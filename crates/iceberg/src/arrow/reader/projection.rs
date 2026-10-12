// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Column projection for `ArrowReader`: building the Parquet projection mask
//! from Iceberg field IDs, and mapping field IDs between Iceberg and Parquet
//! (including fallback handling for files without embedded IDs).

use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use parquet::arrow::{PARQUET_FIELD_ID_META_KEY, ProjectionMask};
use parquet::schema::types::{SchemaDescriptor, Type as ParquetType};

use super::{ArrowReader, CollectFieldIdVisitor};
use crate::arrow::arrow_schema_to_schema;
use crate::error::{Result, invalid_data};
use crate::expr::BoundPredicate;
use crate::expr::visitors::bound_predicate_visitor::visit;
use crate::spec::{NameMapping, NestedField, PrimitiveType, Schema, Type};
use crate::{Error, ErrorKind};

impl ArrowReader {
    pub(super) fn build_field_id_set_and_map(
        parquet_schema: &SchemaDescriptor,
        arrow_schema: &ArrowSchemaRef,
        predicate: &BoundPredicate,
        use_position_fallback: bool,
    ) -> Result<(HashSet<i32>, HashMap<i32, usize>)> {
        // Collects all Iceberg field IDs referenced in the filter predicate
        let mut collector = CollectFieldIdVisitor {
            field_ids: HashSet::default(),
        };
        visit(&mut collector, predicate)?;

        let iceberg_field_ids = collector.field_ids();

        let field_id_map = match build_field_id_map(parquet_schema)? {
            Some(map) => map,
            // No embedded field IDs and no name mapping: position-based fallback
            None if use_position_fallback => build_fallback_field_id_map(parquet_schema),
            // No embedded field IDs, but a name mapping assigned them to the Arrow
            // schema: resolve columns through the mapped Arrow field-id metadata
            None => build_field_id_map_from_arrow_schema(arrow_schema),
        };

        Ok((iceberg_field_ids, field_id_map))
    }

    /// Recursively extract leaf field IDs because Parquet projection works at the leaf column level.
    /// Nested types (struct/list/map) are flattened in Parquet's columnar format.
    fn include_leaf_field_id(field: &NestedField, field_ids: &mut Vec<i32>) {
        match field.field_type.as_ref() {
            Type::Primitive(_) => {
                field_ids.push(field.id);
            }
            Type::Struct(struct_type) => {
                for nested_field in struct_type.fields() {
                    Self::include_leaf_field_id(nested_field, field_ids);
                }
            }
            Type::List(list_type) => {
                Self::include_leaf_field_id(&list_type.element_field, field_ids);
            }
            Type::Map(map_type) => {
                Self::include_leaf_field_id(&map_type.key_field, field_ids);
                Self::include_leaf_field_id(&map_type.value_field, field_ids);
            }
            // Variant projection is rejected earlier (in `get_arrow_projection_mask`); this
            // arm only keeps the match exhaustive. Treat it as a leaf, like a primitive.
            Type::Variant(_) => {
                field_ids.push(field.id);
            }
        }
    }

    pub(super) fn get_arrow_projection_mask(
        field_ids: &[i32],
        iceberg_schema_of_task: &Schema,
        parquet_schema: &SchemaDescriptor,
        arrow_schema: &ArrowSchemaRef,
        use_fallback: bool, // Position-based fallback: file lacks embedded field IDs and no name mapping assigned any
    ) -> Result<ProjectionMask> {
        fn type_promotion_is_valid(
            file_type: Option<&PrimitiveType>,
            projected_type: Option<&PrimitiveType>,
        ) -> bool {
            match (file_type, projected_type) {
                (Some(lhs), Some(rhs)) if lhs == rhs => true,
                (Some(PrimitiveType::Int), Some(PrimitiveType::Long)) => true,
                (Some(PrimitiveType::Float), Some(PrimitiveType::Double)) => true,
                (
                    Some(PrimitiveType::Decimal {
                        precision: file_precision,
                        scale: file_scale,
                    }),
                    Some(PrimitiveType::Decimal {
                        precision: requested_precision,
                        scale: requested_scale,
                    }),
                ) if requested_precision >= file_precision && file_scale == requested_scale => true,
                // Uuid will be store as Fixed(16) in parquet file, so the read back type will be Fixed(16).
                (Some(PrimitiveType::Fixed(16)), Some(PrimitiveType::Uuid)) => true,
                _ => false,
            }
        }

        if field_ids.is_empty() {
            return Ok(ProjectionMask::all());
        }

        // Reading variant columns is not supported yet (see #2188 follow-ups): reject any
        // projection that touches a variant, rather than returning a partial/incorrect batch.
        for field_id in field_ids {
            if let Some(field) = iceberg_schema_of_task.field_by_id(*field_id)
                && type_contains_variant(&field.field_type)
            {
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    "Reading variant columns is not supported yet",
                ));
            }
        }

        if use_fallback {
            // Position-based projection necessary because file lacks embedded field IDs
            Self::get_arrow_projection_mask_fallback(field_ids, parquet_schema)
        } else {
            // Field-ID-based projection using embedded field IDs from Parquet metadata

            // Parquet's columnar format requires leaf-level (not top-level struct/list/map) projection
            let mut leaf_field_ids = vec![];
            for field_id in field_ids {
                let field = iceberg_schema_of_task.field_by_id(*field_id);
                if let Some(field) = field {
                    Self::include_leaf_field_id(field, &mut leaf_field_ids);
                }
            }

            Self::get_arrow_projection_mask_with_field_ids(
                &leaf_field_ids,
                iceberg_schema_of_task,
                parquet_schema,
                arrow_schema,
                type_promotion_is_valid,
            )
        }
    }

    /// Standard projection using embedded field IDs from Parquet metadata.
    /// For iceberg-java compatibility with ParquetSchemaUtil.pruneColumns().
    fn get_arrow_projection_mask_with_field_ids(
        leaf_field_ids: &[i32],
        iceberg_schema_of_task: &Schema,
        parquet_schema: &SchemaDescriptor,
        arrow_schema: &ArrowSchemaRef,
        type_promotion_is_valid: fn(Option<&PrimitiveType>, Option<&PrimitiveType>) -> bool,
    ) -> Result<ProjectionMask> {
        let mut column_map = HashMap::new();
        let fields = arrow_schema.fields();
        // HashSet for O(1) membership checks instead of O(n) slice scans.
        let leaf_field_id_set: HashSet<i32> = leaf_field_ids.iter().copied().collect();
        // A presence leaf is not a requested field, but the type-check schema still
        // has to keep it. Otherwise a map whose value struct lost every child keeps
        // only its key, and `arrow_schema_to_schema` rejects the one-field map.
        let presence = presence_leaves(iceberg_schema_of_task, arrow_schema, &leaf_field_id_set);
        let presence_ids: HashSet<i32> = presence.iter().map(|leaf| leaf.id).collect();

        // Pre-project only the fields that have been selected, possibly avoiding converting
        // some Arrow types that are not yet supported.
        let projected_arrow_schema = ArrowSchema::new_with_metadata(
            fields.filter_leaves(|_, f| {
                parquet_field_id(f).is_some_and(|field_id| {
                    leaf_field_id_set.contains(&field_id) || presence_ids.contains(&field_id)
                })
            }),
            arrow_schema.metadata().clone(),
        );
        let iceberg_schema = arrow_schema_to_schema(&projected_arrow_schema)?;

        fields.filter_leaves(|idx, field| {
            let Some(field_id) = parquet_field_id(field) else {
                return false;
            };

            let iceberg_field = iceberg_schema_of_task.field_by_id(field_id);
            let parquet_iceberg_field = iceberg_schema.field_by_id(field_id);

            if iceberg_field.is_none() || parquet_iceberg_field.is_none() {
                return false;
            }

            if !type_promotion_is_valid(
                parquet_iceberg_field
                    .unwrap()
                    .field_type
                    .as_primitive_type(),
                iceberg_field.unwrap().field_type.as_primitive_type(),
            ) {
                return false;
            }

            column_map.insert(field_id, idx);
            true
        });

        // Schema evolution: new columns may not exist in old Parquet files.
        // RecordBatchTransformer fills those with initial-default or null.
        let mut indices = vec![];
        for field_id in leaf_field_ids {
            if let Some(col_idx) = column_map.get(field_id) {
                indices.push(*col_idx);
            }
        }

        // `presence` was chosen above. Append any leaf the type check did not already
        // keep. `PromotePlan` drops it and copies the struct nulls (#3378).
        let mut selected: HashSet<usize> = indices.iter().copied().collect();
        for leaf in presence {
            if selected.insert(leaf.index) {
                indices.push(leaf.index);
            }
        }

        if indices.is_empty() {
            // Edge case: All requested columns are new (don't exist in file).
            // Project all columns so RecordBatchTransformer has a batch to transform.
            Ok(ProjectionMask::all())
        } else {
            Ok(ProjectionMask::leaves(parquet_schema, indices))
        }
    }

    /// Fallback projection for Parquet files without field IDs.
    /// Uses position-based matching: field ID N → column position N-1.
    /// Projects entire top-level columns (including nested content) for iceberg-java compatibility.
    fn get_arrow_projection_mask_fallback(
        field_ids: &[i32],
        parquet_schema: &SchemaDescriptor,
    ) -> Result<ProjectionMask> {
        // Position-based: field_id N → column N-1 (field IDs are 1-indexed)
        let parquet_root_fields = parquet_schema.root_schema().get_fields();
        let mut root_indices = vec![];

        for field_id in field_ids.iter() {
            let parquet_pos = (*field_id - 1) as usize;

            if parquet_pos < parquet_root_fields.len() {
                root_indices.push(parquet_pos);
            }
            // RecordBatchTransformer adds missing columns with NULL values
        }

        if root_indices.is_empty() {
            Ok(ProjectionMask::all())
        } else {
            Ok(ProjectionMask::roots(parquet_schema, root_indices))
        }
    }
}

/// Whether `field_type` is, or transitively contains, a variant type.
fn type_contains_variant(field_type: &Type) -> bool {
    match field_type {
        Type::Variant(_) => true,
        Type::Struct(s) => s
            .fields()
            .iter()
            .any(|f| type_contains_variant(&f.field_type)),
        Type::List(l) => type_contains_variant(&l.element_field.field_type),
        Type::Map(m) => {
            type_contains_variant(&m.key_field.field_type)
                || type_contains_variant(&m.value_field.field_type)
        }
        Type::Primitive(_) => false,
    }
}

/// Build the map of parquet field id to Parquet column index in the schema.
/// Returns None if the Parquet file doesn't have field IDs embedded (e.g., migrated tables).
pub(super) fn build_field_id_map(
    parquet_schema: &SchemaDescriptor,
) -> Result<Option<HashMap<i32, usize>>> {
    let mut column_map = HashMap::new();

    for (idx, field) in parquet_schema.columns().iter().enumerate() {
        let field_type = field.self_type();
        match field_type {
            ParquetType::PrimitiveType { basic_info, .. } => {
                if !basic_info.has_id() {
                    return Ok(None);
                }
                column_map.insert(basic_info.id(), idx);
            }
            ParquetType::GroupType { .. } => {
                return Err(invalid_data!(
                    "Leaf column in schema should be primitive type but got {field_type:?}"
                ));
            }
        };
    }

    Ok(Some(column_map))
}

/// Finds the Parquet leaf column index carrying `field_id` by its embedded id.
///
/// Unlike [`build_field_id_map`], a leaf without an embedded id does not abort the
/// search -- it is simply skipped. This tolerates files that legitimately mix id-bearing
/// and id-less leaves, e.g. a Variant column whose internal metadata/value leaves are
/// required by the spec to have no field id, alongside a reserved metadata column that
/// does carry its id.
pub(super) fn find_leaf_by_field_id(
    parquet_schema: &SchemaDescriptor,
    field_id: i32,
) -> Option<usize> {
    parquet_schema.columns().iter().position(|col| {
        matches!(
            col.self_type(),
            ParquetType::PrimitiveType { basic_info, .. }
                if basic_info.has_id() && basic_info.id() == field_id
        )
    })
}

/// Build a fallback field ID map for Parquet files without embedded field IDs.
///
/// Returns the number of primitive (leaf) columns in a Parquet type, recursing into groups.
fn leaf_count(ty: &parquet::schema::types::Type) -> usize {
    if ty.is_primitive() {
        1
    } else {
        ty.get_fields().iter().map(|f| leaf_count(f)).sum()
    }
}

/// Builds a mapping from fallback field IDs to leaf column indices for Parquet files
/// without embedded field IDs. Returns entries only for primitive top-level fields.
///
/// Must use top-level field positions (not leaf column positions) to stay consistent
/// with `add_fallback_field_ids_to_arrow_schema`, which assigns ordinal IDs to
/// top-level Arrow fields. Using leaf positions instead would produce wrong indices
/// when nested types (struct/list/map) expand into multiple leaf columns.
///
/// Mirrors iceberg-java's ParquetSchemaUtil.addFallbackIds() which iterates
/// fileSchema.getFields() assigning ordinal IDs to top-level fields.
pub(super) fn build_fallback_field_id_map(
    parquet_schema: &SchemaDescriptor,
) -> HashMap<i32, usize> {
    let mut column_map = HashMap::new();
    let mut leaf_idx = 0;

    for (top_pos, field) in parquet_schema.root_schema().get_fields().iter().enumerate() {
        let field_id = (top_pos + 1) as i32;
        if field.is_primitive() {
            column_map.insert(field_id, leaf_idx);
        }
        leaf_idx += leaf_count(field);
    }

    column_map
}

/// Builds a mapping from field IDs to leaf column indices using the field-id metadata
/// carried by the Arrow schema.
///
/// Used for Parquet files without embedded field IDs when a name mapping has assigned
/// IDs to the Arrow schema (see [`apply_name_mapping_to_arrow_schema`]): the Parquet
/// schema descriptor itself still has no IDs, but the Arrow leaves are flattened in the
/// same depth-first order as Parquet leaf columns, so the Arrow leaf index lines up with
/// the Parquet column index. Columns the mapping did not match carry no field-id
/// metadata and are simply absent from the map.
fn build_field_id_map_from_arrow_schema(arrow_schema: &ArrowSchemaRef) -> HashMap<i32, usize> {
    let mut column_map = HashMap::new();
    arrow_schema.fields().filter_leaves(|idx, field| {
        if let Some(field_id) = parquet_field_id(field) {
            column_map.insert(field_id, idx);
        }
        false
    });
    column_map
}

/// Returns the field ID in `field`'s `PARQUET:field_id` metadata, if it has a valid one.
fn parquet_field_id(field: &Field) -> Option<i32> {
    field
        .metadata()
        .get(PARQUET_FIELD_ID_META_KEY)
        .and_then(|field_id| i32::from_str(field_id).ok())
}

struct FileLeaf {
    index: usize,
    id: i32,
    repetition: u32,
    /// Field ids of the groups that contain this leaf, outer first.
    ancestors: Vec<i32>,
}

struct FileStruct {
    id: i32,
    depth: usize,
}

/// One leaf under each projected struct that would otherwise contribute no column.
///
/// Deepest struct first, so an inner struct's leaf also covers its ancestors.
/// Among a struct's leaves, the lowest repetition depth wins (a direct child
/// rather than one nested in a list or map), then the lowest leaf index.
/// Java: `ParquetSchemaUtil.PresenceColumnSelector`.
fn presence_leaves(
    iceberg_schema: &Schema,
    arrow_schema: &ArrowSchema,
    requested_leaves: &HashSet<i32>,
) -> Vec<FileLeaf> {
    let projected = projected_struct_ids(iceberg_schema, requested_leaves);
    if projected.is_empty() {
        return Vec::new();
    }

    let (leaves, mut structs) = scan_file_fields(arrow_schema.fields());
    structs.retain(|struct_field| projected.contains(&struct_field.id));
    structs.sort_by_key(|struct_field| std::cmp::Reverse(struct_field.depth));

    let mut chosen = Vec::new();
    let mut covered_ids: HashSet<i32> = leaves
        .iter()
        .filter(|leaf| requested_leaves.contains(&leaf.id))
        .map(|leaf| leaf.id)
        .collect();
    for struct_field in structs {
        let mut covered = false;
        let mut best: Option<&FileLeaf> = None;
        for leaf in &leaves {
            if !leaf.ancestors.contains(&struct_field.id) {
                continue;
            }
            if covered_ids.contains(&leaf.id) {
                covered = true;
                break;
            }
            best = Some(match best {
                Some(current)
                    if (current.repetition, current.index) <= (leaf.repetition, leaf.index) =>
                {
                    current
                }
                _ => leaf,
            });
        }
        if covered {
            continue;
        }
        if let Some(leaf) = best {
            covered_ids.insert(leaf.id);
            chosen.push(FileLeaf {
                index: leaf.index,
                id: leaf.id,
                repetition: leaf.repetition,
                ancestors: leaf.ancestors.clone(),
            });
        }
    }
    chosen
}

/// Struct ids that contain a requested leaf. List and map structs are included;
/// the list or map id itself is not a struct.
fn projected_struct_ids(schema: &Schema, requested_leaves: &HashSet<i32>) -> HashSet<i32> {
    let mut out = HashSet::new();
    for field in schema.as_struct().fields() {
        mark_projected_structs(field.as_ref(), requested_leaves, &mut out);
    }
    out
}

fn mark_projected_structs(
    field: &NestedField,
    requested_leaves: &HashSet<i32>,
    out: &mut HashSet<i32>,
) -> bool {
    match field.field_type.as_ref() {
        Type::Primitive(_) | Type::Variant(_) => requested_leaves.contains(&field.id),
        Type::Struct(struct_type) => {
            let mut any = false;
            for child in struct_type.fields() {
                any |= mark_projected_structs(child.as_ref(), requested_leaves, out);
            }
            if any {
                out.insert(field.id);
            }
            any
        }
        Type::List(list) => {
            mark_projected_structs(list.element_field.as_ref(), requested_leaves, out)
        }
        Type::Map(map) => {
            let key = mark_projected_structs(map.key_field.as_ref(), requested_leaves, out);
            let value = mark_projected_structs(map.value_field.as_ref(), requested_leaves, out);
            key || value
        }
    }
}

/// Leaf indexes in the same order as [`Fields::filter_leaves`].
fn scan_file_fields(fields: &Fields) -> (Vec<FileLeaf>, Vec<FileStruct>) {
    let mut leaves = Vec::new();
    let mut structs = Vec::new();
    let mut ancestors = Vec::new();
    let mut index = 0;
    walk_fields(
        fields,
        0,
        &mut ancestors,
        &mut index,
        &mut leaves,
        &mut structs,
    );
    (leaves, structs)
}

fn walk_fields(
    fields: &Fields,
    repetition: u32,
    ancestors: &mut Vec<i32>,
    index: &mut usize,
    leaves: &mut Vec<FileLeaf>,
    structs: &mut Vec<FileStruct>,
) {
    for field in fields {
        walk_field(field, repetition, ancestors, index, leaves, structs);
    }
}

fn walk_field(
    field: &Field,
    repetition: u32,
    ancestors: &mut Vec<i32>,
    index: &mut usize,
    leaves: &mut Vec<FileLeaf>,
    structs: &mut Vec<FileStruct>,
) {
    // One unwrap, matching `Fields::filter_leaves`: a dictionary or run-end
    // encoded column is visited as its value type, and that value is not unwrapped again.
    let logical = match field.data_type() {
        DataType::Dictionary(_, value) => value.as_ref(),
        DataType::RunEndEncoded(_, values) => values.data_type(),
        other => other,
    };
    let field_id = parquet_field_id(field);
    match logical {
        DataType::Struct(children) => {
            if let Some(id) = field_id {
                ancestors.push(id);
                structs.push(FileStruct {
                    id,
                    depth: ancestors.len(),
                });
            }
            walk_fields(children, repetition, ancestors, index, leaves, structs);
            if field_id.is_some() {
                ancestors.pop();
            }
        }
        DataType::List(child) | DataType::LargeList(child) | DataType::FixedSizeList(child, _) => {
            if let Some(id) = field_id {
                ancestors.push(id);
            }
            walk_field(
                child.as_ref(),
                repetition + 1,
                ancestors,
                index,
                leaves,
                structs,
            );
            if field_id.is_some() {
                ancestors.pop();
            }
        }
        DataType::Map(child, _) => {
            if let Some(id) = field_id {
                ancestors.push(id);
            }
            walk_field(
                child.as_ref(),
                repetition + 1,
                ancestors,
                index,
                leaves,
                structs,
            );
            if field_id.is_some() {
                ancestors.pop();
            }
        }
        DataType::Union(union_fields, _) => {
            if let Some(id) = field_id {
                ancestors.push(id);
            }
            for (_, child) in union_fields.iter() {
                walk_field(
                    child.as_ref(),
                    repetition,
                    ancestors,
                    index,
                    leaves,
                    structs,
                );
            }
            if field_id.is_some() {
                ancestors.pop();
            }
        }
        _ => {
            if let Some(id) = field_id {
                leaves.push(FileLeaf {
                    index: *index,
                    id,
                    repetition,
                    ancestors: ancestors.clone(),
                });
            }
            *index += 1;
        }
    }
}

/// Apply name mapping to Arrow schema for Parquet files lacking field IDs.
///
/// Assigns Iceberg field IDs based on column names using the name mapping,
/// enabling correct projection on migrated files (e.g., from Hive/Spark via add_files).
///
/// Per Iceberg spec Column Projection rule #2:
/// "Use schema.name-mapping.default metadata to map field id to columns without field id"
/// https://iceberg.apache.org/spec/#column-projection
///
/// Corresponds to Java's ParquetSchemaUtil.applyNameMapping() and ApplyNameMapping visitor.
/// The key difference is Java operates on Parquet MessageType, while we operate on Arrow Schema.
///
/// # Arguments
/// * `arrow_schema` - Arrow schema from Parquet file (without field IDs)
/// * `name_mapping` - Name mapping from table metadata (TableProperties.DEFAULT_NAME_MAPPING)
///
/// # Returns
/// Arrow schema with field IDs assigned based on name mapping
pub(super) fn apply_name_mapping_to_arrow_schema(
    arrow_schema: ArrowSchemaRef,
    name_mapping: &NameMapping,
) -> Result<Arc<ArrowSchema>> {
    debug_assert!(
        arrow_schema
            .fields()
            .iter()
            .next()
            .is_none_or(|f| f.metadata().get(PARQUET_FIELD_ID_META_KEY).is_none()),
        "Schema already has field IDs - name mapping should not be applied"
    );

    let mut field_ids_by_name = HashMap::with_capacity(name_mapping.fields().len());
    for mapped_field in name_mapping.fields() {
        for name in mapped_field.names() {
            // Preserve the existing first-match behavior for duplicate aliases.
            field_ids_by_name
                .entry(name.as_str())
                .or_insert(mapped_field.field_id());
        }
    }

    let fields_with_mapped_ids: Vec<_> = arrow_schema
        .fields()
        .iter()
        .map(|field| {
            // Look up this column name in name mapping to get the Iceberg field ID.
            // Corresponds to Java's ApplyNameMapping visitor which calls
            // nameMapping.find(currentPath()) and returns field.withId() if found.
            //
            // If the field isn't in the mapping, leave it WITHOUT assigning an ID
            // (matching Java's behavior of returning the field unchanged).
            // Later, during projection, fields without IDs are filtered out.
            let mut metadata = field.metadata().clone();

            if let Some(field_id) = field_ids_by_name
                .get(field.name().as_str())
                .copied()
                .flatten()
            {
                // Field found in mapping with a field_id → assign it
                metadata.insert(PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string());
            }
            // If field_id is None, leave the field without an ID (will be filtered by projection)

            Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                .with_metadata(metadata)
        })
        .collect();

    Ok(Arc::new(ArrowSchema::new_with_metadata(
        fields_with_mapped_ids,
        arrow_schema.metadata().clone(),
    )))
}

/// Add position-based fallback field IDs to Arrow schema for Parquet files lacking them.
/// Enables projection on migrated files (e.g., from Hive/Spark).
///
/// Why at schema level (not per-batch): Efficiency - avoids repeated schema modification.
/// Why only top-level: Nested projection uses leaf column indices, not parent struct IDs.
/// Why 1-indexed: Compatibility with iceberg-java's ParquetSchemaUtil.addFallbackIds().
pub(super) fn add_fallback_field_ids_to_arrow_schema(
    arrow_schema: &ArrowSchemaRef,
) -> Arc<ArrowSchema> {
    debug_assert!(
        arrow_schema
            .fields()
            .iter()
            .next()
            .is_none_or(|f| f.metadata().get(PARQUET_FIELD_ID_META_KEY).is_none()),
        "Schema already has field IDs"
    );

    let fields_with_fallback_ids: Vec<_> = arrow_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(pos, field)| {
            let mut metadata = field.metadata().clone();
            let field_id = (pos + 1) as i32; // 1-indexed for Java compatibility
            metadata.insert(PARQUET_FIELD_ID_META_KEY.to_string(), field_id.to_string());

            Field::new(field.name(), field.data_type().clone(), field.is_nullable())
                .with_metadata(metadata)
        })
        .collect();

    Arc::new(ArrowSchema::new_with_metadata(
        fields_with_fallback_ids,
        arrow_schema.metadata().clone(),
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{Array, ArrayRef, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema, TimeUnit};
    use futures::TryStreamExt;
    use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY, ProjectionMask};
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use parquet::schema::parser::parse_message_type;
    use parquet::schema::types::SchemaDescriptor;
    use tempfile::TempDir;

    use crate::arrow::{ArrowReader, ArrowReaderBuilder};
    use crate::expr::{Bind, Reference};
    use crate::io::FileIO;
    use crate::scan::{FileScanTask, FileScanTaskStream};
    use crate::spec::{
        DataFileFormat, Datum, MappedField, NameMapping, NestedField, PrimitiveType, Schema,
        StructType, Type, VariantType,
    };
    use crate::{ErrorKind, Runtime};

    #[test]
    fn test_name_mapping_duplicate_alias_preserves_first_match() {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "name",
            DataType::Utf8,
            true,
        )]));
        for (first_id, expected) in [(Some(2), Some("2")), (None, None)] {
            let name_mapping = NameMapping::new(vec![
                MappedField::new(first_id, vec!["name".to_string()], vec![]),
                MappedField::new(Some(4), vec!["name".to_string()], vec![]),
            ]);

            let mapped =
                super::apply_name_mapping_to_arrow_schema(arrow_schema.clone(), &name_mapping)
                    .unwrap();

            assert_eq!(
                mapped.fields()[0]
                    .metadata()
                    .get(PARQUET_FIELD_ID_META_KEY)
                    .map(String::as_str),
                expected,
                "first matching field ID: {first_id:?}"
            );
        }
    }

    #[test]
    fn test_arrow_projection_mask() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_identifier_field_ids(vec![1])
                .with_fields(vec![
                    NestedField::required(1, "c1", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(2, "c2", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(
                        3,
                        "c3",
                        Type::Primitive(PrimitiveType::Decimal {
                            precision: 38,
                            scale: 3,
                        }),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("c1", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            // Type not supported
            Field::new("c2", DataType::Duration(TimeUnit::Microsecond), true).with_metadata(
                HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), "2".to_string())]),
            ),
            // Precision is beyond the supported range
            Field::new("c3", DataType::Decimal128(39, 3), true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "3".to_string(),
            )])),
        ]));

        let message_type = "
message schema {
  required binary c1 (STRING) = 1;
  optional int32 c2 (INTEGER(8,true)) = 2;
  optional fixed_len_byte_array(17) c3 (DECIMAL(39,3)) = 3;
}
    ";
        let parquet_type = parse_message_type(message_type).expect("should parse schema");
        let parquet_schema = SchemaDescriptor::new(Arc::new(parquet_type));

        // Try projecting the fields c2 and c3 with the unsupported data types
        let err = ArrowReader::get_arrow_projection_mask(
            &[1, 2, 3],
            &schema,
            &parquet_schema,
            &arrow_schema,
            false,
        )
        .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            err.to_string(),
            "DataInvalid => Unsupported Arrow data type: Duration(µs)".to_string()
        );

        // Omitting field c2, we still get an error due to c3 being selected
        let err = ArrowReader::get_arrow_projection_mask(
            &[1, 3],
            &schema,
            &parquet_schema,
            &arrow_schema,
            false,
        )
        .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::DataInvalid);
        assert_eq!(
            err.to_string(),
            "DataInvalid => Failed to create decimal type, source: DataInvalid => Decimals with precision larger than 38 are not supported: 39".to_string()
        );

        // Finally avoid selecting fields with unsupported data types
        let mask = ArrowReader::get_arrow_projection_mask(
            &[1],
            &schema,
            &parquet_schema,
            &arrow_schema,
            false,
        )
        .expect("Some ProjectionMask");
        assert_eq!(mask, ProjectionMask::leaves(&parquet_schema, vec![0]));
    }

    #[test]
    fn test_arrow_projection_mask_variant_is_unsupported() {
        // Reading variant columns is not supported yet: projecting one (top-level or
        // nested) must fail loudly rather than return a partial/incorrect batch.
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "v", Type::Variant(VariantType)).into(),
                    NestedField::required(
                        3,
                        "s",
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(4, "vv", Type::Variant(VariantType)).into(),
                        ])),
                    )
                    .into(),
                    NestedField::required(
                        5,
                        "m",
                        Type::Map(crate::spec::MapType::required(
                            6,
                            Type::Primitive(PrimitiveType::String),
                            7,
                            Type::Variant(VariantType),
                        )),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        // The parquet/arrow schemas are irrelevant: the variant is rejected before they
        // are consulted, so an empty descriptor is enough to drive the code path.
        let parquet_schema = SchemaDescriptor::new(Arc::new(
            parse_message_type("message schema { optional int32 id = 1; }").unwrap(),
        ));
        let arrow_schema = Arc::new(ArrowSchema::empty());

        // 2 = top-level variant, 3 = struct containing a variant, 4 = the nested variant,
        // 5 = map<string, variant>, plus a mix with a non-variant sibling.
        for projected in [vec![2], vec![3], vec![4], vec![5], vec![1, 2]] {
            let err = ArrowReader::get_arrow_projection_mask(
                &projected,
                &schema,
                &parquet_schema,
                &arrow_schema,
                false,
            )
            .expect_err("variant projection must be rejected");
            assert_eq!(err.kind(), ErrorKind::FeatureUnsupported, "{err}");
        }
    }

    /// Test schema evolution: reading old Parquet file (with only column 'a')
    /// using a newer table schema (with columns 'a' and 'b').
    /// This tests that:
    /// 1. get_arrow_projection_mask allows missing columns
    /// 2. RecordBatchTransformer adds missing column 'b' with NULL values
    #[tokio::test]
    async fn test_schema_evolution_add_column() {
        use arrow_array::{Array, Int32Array};

        // New table schema: columns 'a' and 'b' (b was added later, file only has 'a')
        let new_schema = Arc::new(
            Schema::builder()
                .with_schema_id(2)
                .with_fields(vec![
                    NestedField::required(1, "a", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "b", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        // Create Arrow schema for old Parquet file (only has column 'a')
        let arrow_schema_old = Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
        ]));

        // Write old Parquet file with only column 'a'
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let data_a = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let to_write = RecordBatch::try_new(arrow_schema_old.clone(), vec![data_a]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let file = File::create(format!("{table_location}/old_file.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();
        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        // Read the old Parquet file using the NEW schema (with column 'b')
        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();
        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/old_file.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/old_file.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(new_schema.clone())
                .with_project_field_ids(vec![1, 2]) // Request both columns 'a' and 'b'
                .with_case_sensitive(false)
                .build()
                .unwrap())]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        // Verify we got the correct data
        assert_eq!(result.len(), 1);
        let batch = &result[0];

        // Should have 2 columns now
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 3);

        // Column 'a' should have the original data
        let col_a = batch
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(col_a.values(), &[1, 2, 3]);

        // Column 'b' should be all NULLs (it didn't exist in the old file)
        let col_b = batch
            .column(1)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(col_b.null_count(), 3);
        assert!(col_b.is_null(0));
        assert!(col_b.is_null(1));
        assert!(col_b.is_null(2));
    }

    /// Test reading Parquet files without field ID metadata (e.g., migrated tables).
    /// This exercises the position-based fallback path.
    ///
    /// Corresponds to Java's ParquetSchemaUtil.addFallbackIds() + pruneColumnsFallback()
    /// in /parquet/src/main/java/org/apache/iceberg/parquet/ParquetSchemaUtil.java
    #[tokio::test]
    async fn test_read_parquet_file_without_field_ids() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(2, "age", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        // Parquet file from a migrated table - no field ID metadata
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, false),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let name_data = vec!["Alice", "Bob", "Charlie"];
        let age_data = vec![30, 25, 35];

        use arrow_array::Int32Array;
        let name_col = Arc::new(StringArray::from(name_data.clone())) as ArrayRef;
        let age_col = Arc::new(Int32Array::from(age_data.clone())) as ArrayRef;

        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![name_col, age_col]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();

        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/1.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/1.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1, 2])
                .with_case_sensitive(false)
                .build()
                .unwrap())]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);

        // Verify position-based mapping: field_id 1 → position 0, field_id 2 → position 1
        let name_array = batch.column(0).as_string::<i32>();
        assert_eq!(name_array.value(0), "Alice");
        assert_eq!(name_array.value(1), "Bob");
        assert_eq!(name_array.value(2), "Charlie");

        let age_array = batch
            .column(1)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(age_array.value(0), 30);
        assert_eq!(age_array.value(1), 25);
        assert_eq!(age_array.value(2), 35);
    }

    /// Regression test for #2403: when a Parquet file lacks embedded field IDs but a
    /// name mapping is present, projection must use the field IDs assigned by the name
    /// mapping — not the position-based fallback (field_id N → column N-1).
    ///
    /// The scenario uses a file whose physical column order does not line up with the
    /// Iceberg field IDs: physical columns are `[name, subdept]` while the mapping
    /// assigns `name → 2` and `subdept → 4`. The position fallback would project
    /// field 2 from physical column 1 (`subdept`) and drop field 4 (column 3 is out of
    /// range), silently misreading data.
    #[tokio::test]
    async fn test_read_parquet_with_name_mapping_uses_mapped_field_ids() {
        // Iceberg schema: physical file order does NOT match field-id order.
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(3, "dept", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(4, "subdept", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .unwrap(),
        );

        // Migrated Parquet file: no field-id metadata, physical columns [name, subdept].
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("subdept", DataType::Utf8, true),
        ]));

        let name_mapping = Arc::new(NameMapping::new(vec![
            MappedField::new(Some(2), vec!["name".to_string()], vec![]),
            MappedField::new(Some(4), vec!["subdept".to_string()], vec![]),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let name_col = Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])) as ArrayRef;
        let subdept_col = Arc::new(StringArray::from(vec!["comms", "tax", "audit"])) as ArrayRef;

        let to_write =
            RecordBatch::try_new(arrow_schema.clone(), vec![name_col, subdept_col]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();
        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/1.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/1.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![2, 4])
                .with_case_sensitive(false)
                .with_name_mapping(Some(name_mapping))
                .build()
                .unwrap())]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 2);

        // field 2 (`name`) must come from physical column 0, not be NULL-filled.
        let name_array = batch.column(0).as_string::<i32>();
        assert_eq!(
            name_array.null_count(),
            0,
            "`name` was NULL-filled: name mapping was ignored and position fallback was used"
        );
        assert_eq!(name_array.value(0), "Alice");
        assert_eq!(name_array.value(1), "Bob");
        assert_eq!(name_array.value(2), "Charlie");

        // field 4 (`subdept`) must come from physical column 1.
        let subdept_array = batch.column(1).as_string::<i32>();
        assert_eq!(subdept_array.null_count(), 0);
        assert_eq!(subdept_array.value(0), "comms");
        assert_eq!(subdept_array.value(1), "tax");
        assert_eq!(subdept_array.value(2), "audit");
    }

    /// Regression test for #2403, predicate side: with a name mapping present, predicate
    /// pushdown must resolve field IDs via the mapping rather than the position fallback,
    /// which would evaluate the filter against the wrong physical column.
    #[tokio::test]
    async fn test_predicate_on_name_mapped_file_uses_mapped_field_ids() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(3, "dept", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::optional(4, "subdept", Type::Primitive(PrimitiveType::String))
                        .into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("subdept", DataType::Utf8, true),
        ]));

        let name_mapping = Arc::new(NameMapping::new(vec![
            MappedField::new(Some(2), vec!["name".to_string()], vec![]),
            MappedField::new(Some(4), vec!["subdept".to_string()], vec![]),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        // Engineered so that filtering the wrong physical column yields different rows:
        // `name` and `subdept` both contain the value "Alice", on different rows.
        let name_col = Arc::new(StringArray::from(vec!["Alice", "Bob", "Sue"])) as ArrayRef;
        let subdept_col = Arc::new(StringArray::from(vec!["Bob", "Alice", "Alice"])) as ArrayRef;

        let to_write =
            RecordBatch::try_new(arrow_schema.clone(), vec![name_col, subdept_col]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();
        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let predicate = Reference::new("name").equal_to(Datum::string("Alice"));

        let reader = ArrowReaderBuilder::new(file_io, Runtime::current())
            .with_row_group_filtering_enabled(true)
            .with_row_selection_enabled(true)
            .build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/1.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/1.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![2, 4])
                .with_case_sensitive(false)
                .with_name_mapping(Some(name_mapping))
                .with_predicate(Some(predicate.bind(schema, true).unwrap()))
                .build()
                .unwrap())]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            total_rows, 1,
            "filter `name = \"Alice\"` matched the wrong rows: predicate was evaluated \
             against the wrong physical column"
        );

        let batch = &result[0];
        let name_array = batch.column(0).as_string::<i32>();
        assert_eq!(name_array.value(0), "Alice");
        let subdept_array = batch.column(1).as_string::<i32>();
        assert_eq!(subdept_array.value(0), "Bob");
    }

    /// Test reading Parquet files without field IDs with partial projection.
    /// Only a subset of columns are requested, verifying position-based fallback
    /// handles column selection correctly.
    #[tokio::test]
    async fn test_read_parquet_without_field_ids_partial_projection() {
        use arrow_array::Int32Array;

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "col1", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(2, "col2", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(3, "col3", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(4, "col4", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("col1", DataType::Utf8, false),
            Field::new("col2", DataType::Int32, false),
            Field::new("col3", DataType::Utf8, false),
            Field::new("col4", DataType::Int32, false),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let col1_data = Arc::new(StringArray::from(vec!["a", "b"])) as ArrayRef;
        let col2_data = Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef;
        let col3_data = Arc::new(StringArray::from(vec!["c", "d"])) as ArrayRef;
        let col4_data = Arc::new(Int32Array::from(vec![30, 40])) as ArrayRef;

        let to_write = RecordBatch::try_new(arrow_schema.clone(), vec![
            col1_data, col2_data, col3_data, col4_data,
        ])
        .unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();

        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/1.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/1.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1, 3])
                .with_case_sensitive(false)
                .build()
                .unwrap())]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);

        let col1_array = batch.column(0).as_string::<i32>();
        assert_eq!(col1_array.value(0), "a");
        assert_eq!(col1_array.value(1), "b");

        let col3_array = batch.column(1).as_string::<i32>();
        assert_eq!(col3_array.value(0), "c");
        assert_eq!(col3_array.value(1), "d");
    }

    /// Test reading Parquet files without field IDs with schema evolution.
    /// The Iceberg schema has more fields than the Parquet file, testing that
    /// missing columns are filled with NULLs.
    #[tokio::test]
    async fn test_read_parquet_without_field_ids_schema_evolution() {
        use arrow_array::{Array, Int32Array};

        // Schema with field 3 added after the file was written
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(2, "age", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(3, "city", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, false),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let name_data = Arc::new(StringArray::from(vec!["Alice", "Bob"])) as ArrayRef;
        let age_data = Arc::new(Int32Array::from(vec![30, 25])) as ArrayRef;

        let to_write =
            RecordBatch::try_new(arrow_schema.clone(), vec![name_data, age_data]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();

        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/1.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/1.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1, 2, 3])
                .with_case_sensitive(false)
                .build()
                .unwrap())]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 3);

        let name_array = batch.column(0).as_string::<i32>();
        assert_eq!(name_array.value(0), "Alice");
        assert_eq!(name_array.value(1), "Bob");

        let age_array = batch
            .column(1)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(age_array.value(0), 30);
        assert_eq!(age_array.value(1), 25);

        // Verify missing column filled with NULLs
        let city_array = batch.column(2).as_string::<i32>();
        assert_eq!(city_array.null_count(), 2);
        assert!(city_array.is_null(0));
        assert!(city_array.is_null(1));
    }

    /// Test reading Parquet files without field IDs that have multiple row groups.
    /// This ensures the position-based fallback works correctly across row group boundaries.
    #[tokio::test]
    async fn test_read_parquet_without_field_ids_multiple_row_groups() {
        use arrow_array::Int32Array;

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(2, "value", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        // Small row group size to create multiple row groups
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_write_batch_size(2)
            .set_max_row_group_row_count(Some(2))
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props)).unwrap();

        // Write 6 rows in 3 batches (will create 3 row groups)
        for batch_num in 0..3 {
            let name_data = Arc::new(StringArray::from(vec![
                format!("name_{}", batch_num * 2),
                format!("name_{}", batch_num * 2 + 1),
            ])) as ArrayRef;
            let value_data =
                Arc::new(Int32Array::from(vec![batch_num * 2, batch_num * 2 + 1])) as ArrayRef;

            let batch =
                RecordBatch::try_new(arrow_schema.clone(), vec![name_data, value_data]).unwrap();
            writer.write(&batch).expect("Writing batch");
        }
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/1.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/1.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1, 2])
                .with_case_sensitive(false)
                .build()
                .unwrap())]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert!(!result.is_empty());

        let mut all_names = Vec::new();
        let mut all_values = Vec::new();

        for batch in &result {
            let name_array = batch.column(0).as_string::<i32>();
            let value_array = batch
                .column(1)
                .as_primitive::<arrow_array::types::Int32Type>();

            for i in 0..batch.num_rows() {
                all_names.push(name_array.value(i).to_string());
                all_values.push(value_array.value(i));
            }
        }

        assert_eq!(all_names.len(), 6);
        assert_eq!(all_values.len(), 6);

        for i in 0..6 {
            assert_eq!(all_names[i], format!("name_{i}"));
            assert_eq!(all_values[i], i as i32);
        }
    }

    /// Test reading Parquet files without field IDs with nested types (struct).
    /// Java's pruneColumnsFallback() projects entire top-level columns including nested content.
    /// This test verifies that a top-level struct field is projected correctly with all its nested fields.
    #[tokio::test]
    async fn test_read_parquet_without_field_ids_with_struct() {
        use arrow_array::{Int32Array, StructArray};
        use arrow_schema::Fields;

        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(
                        2,
                        "person",
                        Type::Struct(StructType::new(vec![
                            NestedField::required(
                                3,
                                "name",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                            NestedField::required(4, "age", Type::Primitive(PrimitiveType::Int))
                                .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "person",
                DataType::Struct(Fields::from(vec![
                    Field::new("name", DataType::Utf8, false),
                    Field::new("age", DataType::Int32, false),
                ])),
                false,
            ),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let id_data = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
        let name_data = Arc::new(StringArray::from(vec!["Alice", "Bob"])) as ArrayRef;
        let age_data = Arc::new(Int32Array::from(vec![30, 25])) as ArrayRef;
        let person_data = Arc::new(StructArray::from(vec![
            (
                Arc::new(Field::new("name", DataType::Utf8, false)),
                name_data,
            ),
            (
                Arc::new(Field::new("age", DataType::Int32, false)),
                age_data,
            ),
        ])) as ArrayRef;

        let to_write =
            RecordBatch::try_new(arrow_schema.clone(), vec![id_data, person_data]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();

        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/1.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/1.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1, 2])
                .with_case_sensitive(false)
                .build()
                .unwrap())]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);

        let id_array = batch
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(id_array.value(0), 1);
        assert_eq!(id_array.value(1), 2);

        let person_array = batch.column(1).as_struct();
        assert_eq!(person_array.num_columns(), 2);

        let name_array = person_array.column(0).as_string::<i32>();
        assert_eq!(name_array.value(0), "Alice");
        assert_eq!(name_array.value(1), "Bob");

        let age_array = person_array
            .column(1)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(age_array.value(0), 30);
        assert_eq!(age_array.value(1), 25);
    }

    /// Test reading Parquet files without field IDs with schema evolution - column added in the middle.
    /// When a new column is inserted between existing columns in the schema order,
    /// the fallback projection must correctly map field IDs to output positions.
    #[tokio::test]
    async fn test_read_parquet_without_field_ids_schema_evolution_add_column_in_middle() {
        use arrow_array::{Array, Int32Array};

        let arrow_schema_old = Arc::new(ArrowSchema::new(vec![
            Field::new("col0", DataType::Int32, true),
            Field::new("col1", DataType::Int32, true),
        ]));

        // New column added between existing columns: col0 (id=1), newCol (id=5), col1 (id=2)
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::optional(1, "col0", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(5, "newCol", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "col1", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let col0_data = Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef;
        let col1_data = Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef;

        let to_write =
            RecordBatch::try_new(arrow_schema_old.clone(), vec![col0_data, col1_data]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();
        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/1.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/1.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1, 5, 2])
                .with_case_sensitive(false)
                .build()
                .unwrap())]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let batch = &result[0];
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 3);

        let result_col0 = batch
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(result_col0.value(0), 1);
        assert_eq!(result_col0.value(1), 2);

        // New column should be NULL (doesn't exist in old file)
        let result_newcol = batch
            .column(1)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(result_newcol.null_count(), 2);
        assert!(result_newcol.is_null(0));
        assert!(result_newcol.is_null(1));

        let result_col1 = batch
            .column(2)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(result_col1.value(0), 10);
        assert_eq!(result_col1.value(1), 20);
    }

    /// Test reading Parquet files without field IDs with a filter that eliminates all row groups.
    /// During development of field ID mapping, we saw a panic when row_selection_enabled=true and
    /// all row groups are filtered out.
    #[tokio::test]
    async fn test_read_parquet_without_field_ids_filter_eliminates_all_rows() {
        use arrow_array::{Float64Array, Int32Array};

        // Schema with fields that will use fallback IDs 1, 2, 3
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                    NestedField::required(3, "value", Type::Primitive(PrimitiveType::Double))
                        .into(),
                ])
                .build()
                .unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Float64, false),
        ]));

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        // Write data where all ids are >= 10
        let id_data = Arc::new(Int32Array::from(vec![10, 11, 12])) as ArrayRef;
        let name_data = Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef;
        let value_data = Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])) as ArrayRef;

        let to_write =
            RecordBatch::try_new(arrow_schema.clone(), vec![id_data, name_data, value_data])
                .unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let file = File::create(format!("{table_location}/1.parquet")).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();
        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        // Filter that eliminates all row groups: id < 5
        let predicate = Reference::new("id").less_than(Datum::int(5));

        // Enable both row_group_filtering and row_selection - triggered the panic
        let reader = ArrowReaderBuilder::new(file_io, Runtime::current())
            .with_row_group_filtering_enabled(true)
            .with_row_selection_enabled(true)
            .build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/1.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/1.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1, 2, 3])
                .with_case_sensitive(false)
                .with_predicate(Some(predicate.bind(schema, true).unwrap()))
                .build()
                .unwrap())]
            .into_iter(),
        )) as FileScanTaskStream;

        // Should no longer panic
        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        // Should return empty results
        assert!(result.is_empty() || result.iter().all(|batch| batch.num_rows() == 0));
    }

    /// Test bucket partitioning reads source column from data file (not partition metadata).
    ///
    /// This is an integration test verifying the complete ArrowReader pipeline with bucket partitioning.
    /// It corresponds to TestRuntimeFiltering tests in Iceberg Java (e.g., testRenamedSourceColumnTable).
    ///
    /// # Iceberg Spec Requirements
    ///
    /// Per the Iceberg spec "Column Projection" section:
    /// > "Return the value from partition metadata if an **Identity Transform** exists for the field"
    ///
    /// This means:
    /// - Identity transforms (e.g., `identity(dept)`) use constants from partition metadata
    /// - Non-identity transforms (e.g., `bucket(4, id)`) must read source columns from data files
    /// - Partition metadata for bucket transforms stores bucket numbers (0-3), NOT source values
    ///
    /// Java's PartitionUtil.constantsMap() implements this via:
    /// ```java
    /// if (field.transform().isIdentity()) {
    ///     idToConstant.put(field.sourceId(), converted);
    /// }
    /// ```
    ///
    /// # What This Test Verifies
    ///
    /// This test ensures the full ArrowReader → RecordBatchTransformer pipeline correctly handles
    /// bucket partitioning when FileScanTask provides partition_spec and partition_data:
    ///
    /// - Parquet file has field_id=1 named "id" with actual data [1, 5, 9, 13]
    /// - FileScanTask specifies partition_spec with bucket(4, id) and partition_data with bucket=1
    /// - RecordBatchTransformer.constants_map() excludes bucket-partitioned field from constants
    /// - ArrowReader correctly reads [1, 5, 9, 13] from the data file
    /// - Values are NOT replaced with constant 1 from partition metadata
    ///
    /// # Why This Matters
    ///
    /// Without correct handling:
    /// - Runtime filtering would break (e.g., `WHERE id = 5` would fail)
    /// - Query results would be incorrect (all rows would have id=1)
    /// - Bucket partitioning would be unusable for query optimization
    ///
    /// # References
    /// - Iceberg spec: format/spec.md "Column Projection" + "Partition Transforms"
    /// - Java test: spark/src/test/java/.../TestRuntimeFiltering.java
    /// - Java impl: core/src/main/java/org/apache/iceberg/util/PartitionUtil.java
    #[tokio::test]
    async fn test_bucket_partitioning_reads_source_column_from_file() {
        use arrow_array::Int32Array;

        use crate::spec::{Literal, PartitionSpec, Struct, Transform};

        // Iceberg schema with id and name columns
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );

        // Partition spec: bucket(4, id)
        let partition_spec = Arc::new(
            PartitionSpec::builder(schema.clone())
                .with_spec_id(0)
                .add_partition_field("id", "id_bucket", Transform::Bucket(4))
                .unwrap()
                .build()
                .unwrap(),
        );

        // Partition data: bucket value is 1
        let partition_data = Struct::from_iter(vec![Some(Literal::int(1))]);

        // Create Arrow schema with field IDs for Parquet file
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "1".to_string(),
            )])),
            Field::new("name", DataType::Utf8, true).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                "2".to_string(),
            )])),
        ]));

        // Write Parquet file with data
        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_io = FileIO::new_with_fs();

        let id_data = Arc::new(Int32Array::from(vec![1, 5, 9, 13])) as ArrayRef;
        let name_data =
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie", "Dave"])) as ArrayRef;

        let to_write =
            RecordBatch::try_new(arrow_schema.clone(), vec![id_data, name_data]).unwrap();

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let file = File::create(format!("{}/data.parquet", &table_location)).unwrap();
        let mut writer = ArrowWriter::try_new(file, to_write.schema(), Some(props)).unwrap();
        writer.write(&to_write).expect("Writing batch");
        writer.close().unwrap();

        // Read the Parquet file with partition spec and data
        let reader = ArrowReaderBuilder::new(file_io, Runtime::current()).build();
        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask::builder()
                .with_file_size_in_bytes(
                    std::fs::metadata(format!("{table_location}/data.parquet"))
                        .unwrap()
                        .len(),
                )
                .with_start(0)
                .with_length(0)
                .with_data_file_path(format!("{table_location}/data.parquet"))
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(schema.clone())
                .with_project_field_ids(vec![1, 2])
                .with_case_sensitive(false)
                .with_partition(Some(partition_data))
                .with_partition_spec(Some(partition_spec))
                .build()
                .unwrap())]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        // Verify we got the correct data
        assert_eq!(result.len(), 1);
        let batch = &result[0];

        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 4);

        // The id column MUST contain actual values from the Parquet file [1, 5, 9, 13],
        // NOT the constant partition value 1
        let id_col = batch
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(id_col.value(0), 1);
        assert_eq!(id_col.value(1), 5);
        assert_eq!(id_col.value(2), 9);
        assert_eq!(id_col.value(3), 13);

        let name_col = batch.column(1).as_string::<i32>();
        assert_eq!(name_col.value(0), "Alice");
        assert_eq!(name_col.value(1), "Bob");
        assert_eq!(name_col.value(2), "Charlie");
        assert_eq!(name_col.value(3), "Dave");
    }

    /// Regression for <https://github.com/apache/iceberg-rust/issues/2306>:
    /// predicate on a column after nested types in a migrated file (no field IDs).
    /// Schema has struct, list, and map columns before the predicate target (`id`),
    /// exercising the fallback field ID mapping across all nested type variants.
    #[tokio::test]
    async fn test_predicate_on_migrated_file_with_nested_types() {
        use serde::{Deserialize, Serialize};
        use serde_arrow::schema::{SchemaLike, TracingOptions};

        #[derive(Serialize, Deserialize)]
        struct Person {
            name: String,
            age: i32,
        }

        #[derive(Serialize, Deserialize)]
        struct Row {
            person: Person,
            people: Vec<Person>,
            props: std::collections::BTreeMap<String, String>,
            id: i32,
        }

        let rows = vec![
            Row {
                person: Person {
                    name: "Alice".into(),
                    age: 30,
                },
                people: vec![Person {
                    name: "Alice".into(),
                    age: 30,
                }],
                props: [("k1".into(), "v1".into())].into(),
                id: 1,
            },
            Row {
                person: Person {
                    name: "Bob".into(),
                    age: 25,
                },
                people: vec![Person {
                    name: "Bob".into(),
                    age: 25,
                }],
                props: [("k2".into(), "v2".into())].into(),
                id: 2,
            },
            Row {
                person: Person {
                    name: "Carol".into(),
                    age: 40,
                },
                people: vec![Person {
                    name: "Carol".into(),
                    age: 40,
                }],
                props: [("k3".into(), "v3".into())].into(),
                id: 3,
            },
        ];

        let tracing_options = TracingOptions::default()
            .map_as_struct(false)
            .strings_as_large_utf8(false)
            .sequence_as_large_list(false);
        let fields = Vec::<arrow_schema::FieldRef>::from_type::<Row>(tracing_options).unwrap();
        let arrow_schema = Arc::new(ArrowSchema::new(fields.clone()));
        let batch = serde_arrow::to_record_batch(&fields, &rows).unwrap();

        // Fallback field IDs: person=1, people=2, props=3, id=4
        let iceberg_schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(
                        1,
                        "person",
                        Type::Struct(StructType::new(vec![
                            NestedField::required(
                                5,
                                "name",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                            NestedField::required(6, "age", Type::Primitive(PrimitiveType::Int))
                                .into(),
                        ])),
                    )
                    .into(),
                    NestedField::required(
                        2,
                        "people",
                        Type::List(crate::spec::ListType {
                            element_field: NestedField::required(
                                7,
                                "element",
                                Type::Struct(StructType::new(vec![
                                    NestedField::required(
                                        8,
                                        "name",
                                        Type::Primitive(PrimitiveType::String),
                                    )
                                    .into(),
                                    NestedField::required(
                                        9,
                                        "age",
                                        Type::Primitive(PrimitiveType::Int),
                                    )
                                    .into(),
                                ])),
                            )
                            .into(),
                        }),
                    )
                    .into(),
                    NestedField::required(
                        3,
                        "props",
                        Type::Map(crate::spec::MapType {
                            key_field: NestedField::required(
                                10,
                                "key",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                            value_field: NestedField::required(
                                11,
                                "value",
                                Type::Primitive(PrimitiveType::String),
                            )
                            .into(),
                        }),
                    )
                    .into(),
                    NestedField::required(4, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );

        let tmp_dir = TempDir::new().unwrap();
        let table_location = tmp_dir.path().to_str().unwrap().to_string();
        let file_path = format!("{table_location}/1.parquet");

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let file = File::create(&file_path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema, Some(props)).unwrap();
        writer.write(&batch).expect("Writing batch");
        writer.close().unwrap();

        let predicate = Reference::new("id").greater_than(Datum::int(1));

        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current())
            .with_row_group_filtering_enabled(true)
            .with_row_selection_enabled(true)
            .build();

        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask::builder()
                .with_file_size_in_bytes(std::fs::metadata(&file_path).unwrap().len())
                .with_start(0)
                .with_length(0)
                .with_data_file_path(file_path)
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(iceberg_schema.clone())
                .with_project_field_ids(vec![4])
                .with_case_sensitive(false)
                .with_predicate(Some(predicate.bind(iceberg_schema, true).unwrap()))
                .build()
                .unwrap())]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();

        let ids: Vec<i32> = result
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(ids, vec![2, 3]);
    }

    /// Reads a file written with an older nested schema through an evolved one. This relies
    /// on parquet-rs attaching `PARQUET:field_id` to struct children, list elements, and map
    /// keys and values, which the hand-built fixtures in `record_batch_transformer` assume.
    #[tokio::test]
    async fn test_read_evolved_nested_fields_by_field_id() {
        use serde_json::{Value, json};

        use crate::arrow::schema_to_arrow_schema;
        use crate::spec::PrimitiveType::{Int, Long};
        use crate::spec::{ListType, MapType};

        fn struct_of(fields: &[(i32, &str, PrimitiveType)]) -> Type {
            Type::Struct(StructType::new(
                fields
                    .iter()
                    .map(|(id, name, ty)| {
                        NestedField::optional(*id, *name, Type::Primitive(ty.clone())).into()
                    })
                    .collect(),
            ))
        }
        fn schema(s: Type, element: Type, value: Type) -> Schema {
            let string = Type::Primitive(PrimitiveType::String);
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(Int)).into(),
                    NestedField::optional(2, "s", s).into(),
                    NestedField::optional(
                        3,
                        "l",
                        Type::List(ListType::new(
                            NestedField::list_element(7, element, true).into(),
                        )),
                    )
                    .into(),
                    NestedField::optional(
                        4,
                        "m",
                        Type::Map(MapType::required(9, string, 10, value)),
                    )
                    .into(),
                ])
                .build()
                .unwrap()
        }
        fn to_batch(schema: &Schema, rows: Value) -> RecordBatch {
            let fields = schema_to_arrow_schema(schema).unwrap().fields().to_vec();
            serde_arrow::to_record_batch(&fields, &rows).unwrap()
        }

        let old_schema = schema(
            struct_of(&[(5, "a", Int), (6, "b", PrimitiveType::String)]),
            struct_of(&[(8, "x", Int)]),
            struct_of(&[(11, "x", Int)]),
        );
        // Reorders `s`, adds children before, between, and after existing ones, and
        // promotes int to long.
        let new_schema = Arc::new(schema(
            struct_of(&[
                (6, "b", PrimitiveType::String),
                (12, "c", Int),
                (5, "a", Long),
            ]),
            struct_of(&[(13, "y", Int), (8, "x", Int)]),
            struct_of(&[(11, "x", Long), (14, "y", Int)]),
        ));
        let old_batch = to_batch(
            &old_schema,
            json!([
                {"id": 1, "s": {"a": 1, "b": "p"}, "l": [{"x": 10}, {"x": 11}], "m": {"k1": {"x": 100}}},
                {"id": 2, "s": null, "l": null, "m": {"k2": {"x": 200}}},
                {"id": 3, "s": {"a": 3, "b": null}, "l": [{"x": 30}], "m": null},
            ]),
        );
        let expected = to_batch(
            &new_schema,
            json!([
                {
                    "id": 1,
                    "s": {"b": "p", "c": null, "a": 1},
                    "l": [{"y": null, "x": 10}, {"y": null, "x": 11}],
                    "m": {"k1": {"x": 100, "y": null}},
                },
                {"id": 2, "s": null, "l": null, "m": {"k2": {"x": 200, "y": null}}},
                {"id": 3, "s": {"b": null, "c": null, "a": 3}, "l": [{"y": null, "x": 30}], "m": null},
            ]),
        );

        let tmp_dir = TempDir::new().unwrap();
        let file_path = format!("{}/old.parquet", tmp_dir.path().to_str().unwrap());
        let mut writer =
            ArrowWriter::try_new(File::create(&file_path).unwrap(), old_batch.schema(), None)
                .unwrap();
        writer.write(&old_batch).unwrap();
        writer.close().unwrap();

        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current()).build();
        let tasks = Box::pin(futures::stream::iter(
            vec![Ok(FileScanTask::builder()
                .with_file_size_in_bytes(std::fs::metadata(&file_path).unwrap().len())
                .with_start(0)
                .with_length(0)
                .with_data_file_path(file_path)
                .with_data_file_format(DataFileFormat::Parquet)
                .with_schema(new_schema)
                .with_project_field_ids(vec![1, 2, 3, 4])
                .with_case_sensitive(false)
                .build()
                .unwrap())]
            .into_iter(),
        )) as FileScanTaskStream;

        let result = reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect::<Vec<RecordBatch>>()
            .await
            .unwrap();
        assert_eq!(result, vec![expected]);
    }

    fn field_id(name: &str, data_type: DataType, nullable: bool, id: i32) -> Field {
        Field::new(name, data_type, nullable).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            id.to_string(),
        )]))
    }

    fn rows_to_batch(schema: &Schema, rows: serde_json::Value) -> RecordBatch {
        let fields = crate::arrow::schema_to_arrow_schema(schema)
            .unwrap()
            .fields()
            .to_vec();
        serde_arrow::to_record_batch(&fields, &rows).unwrap()
    }

    fn write_parquet(batch: &RecordBatch) -> (TempDir, String) {
        let tmp_dir = TempDir::new().unwrap();
        let path = format!("{}/rows.parquet", tmp_dir.path().to_str().unwrap());
        let mut writer =
            ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), None).unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
        (tmp_dir, path)
    }

    async fn read_parquet(
        path: &str,
        schema: Arc<Schema>,
        project_field_ids: Vec<i32>,
    ) -> Vec<RecordBatch> {
        let reader = ArrowReaderBuilder::new(FileIO::new_with_fs(), Runtime::current()).build();
        let tasks = Box::pin(futures::stream::iter(vec![Ok(FileScanTask::builder()
            .with_file_size_in_bytes(std::fs::metadata(path).unwrap().len())
            .with_start(0)
            .with_length(0)
            .with_data_file_path(path.to_string())
            .with_data_file_format(DataFileFormat::Parquet)
            .with_schema(schema)
            .with_project_field_ids(project_field_ids)
            .with_case_sensitive(false)
            .build()
            .unwrap())])) as FileScanTaskStream;
        reader
            .read(tasks)
            .unwrap()
            .stream()
            .try_collect()
            .await
            .unwrap()
    }

    #[test]
    fn test_projection_mask_reads_one_presence_leaf() {
        // `s` is struct<5: a int, 6: b list<7: int>>. The table replaced both children.
        // Presence must be `a` (repetition 0), not the list element and not every column.
        let file_arrow = Arc::new(ArrowSchema::new(vec![
            field_id("id", DataType::Int32, false, 1),
            field_id(
                "s",
                DataType::Struct(
                    vec![
                        field_id("a", DataType::Int32, true, 5),
                        field_id(
                            "b",
                            DataType::List(Arc::new(field_id("element", DataType::Int32, true, 7))),
                            true,
                            6,
                        ),
                    ]
                    .into(),
                ),
                true,
                2,
            ),
            field_id("other", DataType::Int32, true, 3),
        ]));
        let table = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(
                        2,
                        "s",
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(9, "n", Type::Primitive(PrimitiveType::Int))
                                .into(),
                        ])),
                    )
                    .into(),
                    NestedField::optional(3, "other", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );
        let parquet_schema = SchemaDescriptor::new(Arc::new(
            parse_message_type(
                "
                message schema {
                  required int32 id = 1;
                  optional group s = 2 {
                    optional int32 a = 5;
                    optional group b (LIST) = 6 {
                      repeated group list {
                        optional int32 element = 7;
                      }
                    }
                  }
                  optional int32 other = 3;
                }
                ",
            )
            .unwrap(),
        ));

        let mask = |ids: &[i32]| {
            ArrowReader::get_arrow_projection_mask(ids, &table, &parquet_schema, &file_arrow, false)
                .unwrap()
        };
        assert_eq!(
            mask(&[1, 2]),
            ProjectionMask::leaves(&parquet_schema, vec![0, 1])
        );
        assert_eq!(mask(&[2]), ProjectionMask::leaves(&parquet_schema, vec![1]));
        assert_eq!(mask(&[1]), ProjectionMask::leaves(&parquet_schema, vec![0]));
        assert_eq!(mask(&[3]), ProjectionMask::leaves(&parquet_schema, vec![3]));
    }

    #[tokio::test]
    async fn test_struct_presence_when_requested_children_missing() {
        fn struct_of(id: i32, name: &str) -> Type {
            Type::Struct(StructType::new(vec![
                NestedField::optional(id, name, Type::Primitive(PrimitiveType::Int)).into(),
            ]))
        }
        let file_schema = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "s", struct_of(5, "x")).into(),
            ])
            .build()
            .unwrap();
        let table_schema = Arc::new(
            Schema::builder()
                .with_schema_id(2)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "s", struct_of(6, "x")).into(),
                ])
                .build()
                .unwrap(),
        );
        // Row 0: struct present, child null. Row 1: struct null. Row 2: both present.
        let file_batch = rows_to_batch(
            &file_schema,
            serde_json::json!([
                {"id": 1, "s": {"x": null}},
                {"id": 2, "s": null},
                {"id": 3, "s": {"x": 30}},
            ]),
        );
        let (_dir, path) = write_parquet(&file_batch);
        for project in [vec![2], vec![1, 2]] {
            let projected = Schema::builder()
                .with_schema_id(2)
                .with_fields(
                    project
                        .iter()
                        .map(|id| Arc::clone(table_schema.field_by_id(*id).unwrap()))
                        .collect::<Vec<_>>(),
                )
                .build()
                .unwrap();
            let rows = if project.len() == 1 {
                serde_json::json!([
                    {"s": {"x": null}},
                    {"s": null},
                    {"s": {"x": null}},
                ])
            } else {
                serde_json::json!([
                    {"id": 1, "s": {"x": null}},
                    {"id": 2, "s": null},
                    {"id": 3, "s": {"x": null}},
                ])
            };
            let expected = rows_to_batch(&projected, rows);
            let out = read_parquet(&path, table_schema.clone(), project.clone()).await;
            assert_eq!(out, vec![expected], "projection {project:?}");
        }
    }

    #[tokio::test]
    async fn test_struct_presence_applies_initial_default() {
        use crate::spec::Literal;

        let file_schema = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(
                    2,
                    "s",
                    Type::Struct(StructType::new(vec![
                        NestedField::optional(5, "x", Type::Primitive(PrimitiveType::Int)).into(),
                    ])),
                )
                .into(),
            ])
            .build()
            .unwrap();
        let table_schema = Arc::new(
            Schema::builder()
                .with_schema_id(2)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(
                        2,
                        "s",
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(6, "y", Type::Primitive(PrimitiveType::Int))
                                .with_initial_default(Literal::int(0))
                                .into(),
                        ])),
                    )
                    .into(),
                ])
                .build()
                .unwrap(),
        );
        let file_batch = rows_to_batch(
            &file_schema,
            serde_json::json!([
                {"id": 1, "s": {"x": 10}},
                {"id": 2, "s": null},
                {"id": 3, "s": {"x": null}},
            ]),
        );
        let expected = rows_to_batch(
            &table_schema,
            serde_json::json!([
                {"id": 1, "s": {"y": 0}},
                {"id": 2, "s": null},
                {"id": 3, "s": {"y": 0}},
            ]),
        );
        let (_dir, path) = write_parquet(&file_batch);
        let out = read_parquet(&path, table_schema, vec![1, 2]).await;
        assert_eq!(out, vec![expected]);
    }

    #[tokio::test]
    async fn test_nested_list_and_map_struct_presence() {
        use crate::spec::{ListType, MapType};

        fn inner(id: i32) -> Type {
            Type::Struct(StructType::new(vec![
                NestedField::optional(id, "y", Type::Primitive(PrimitiveType::Int)).into(),
            ]))
        }
        fn element(id: i32) -> Type {
            Type::Struct(StructType::new(vec![
                NestedField::optional(id, "x", Type::Primitive(PrimitiveType::Int)).into(),
            ]))
        }
        fn schema(inner_id: i32, element_id: i32, value_id: i32) -> Schema {
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(
                        2,
                        "s",
                        Type::Struct(StructType::new(vec![
                            NestedField::optional(5, "a", Type::Primitive(PrimitiveType::Int))
                                .into(),
                            NestedField::optional(7, "inner", inner(inner_id)).into(),
                        ])),
                    )
                    .into(),
                    NestedField::optional(
                        3,
                        "l",
                        Type::List(ListType::new(
                            NestedField::list_element(10, element(element_id), false).into(),
                        )),
                    )
                    .into(),
                    NestedField::optional(
                        4,
                        "m",
                        Type::Map(MapType::optional(
                            13,
                            Type::Primitive(PrimitiveType::String),
                            14,
                            element(value_id),
                        )),
                    )
                    .into(),
                ])
                .build()
                .unwrap()
        }

        let file_schema = schema(8, 11, 15);
        let table_schema = Arc::new(schema(9, 12, 16));
        let file_batch = rows_to_batch(
            &file_schema,
            serde_json::json!([
                {
                    "id": 1,
                    "s": {"a": 1, "inner": {"y": 5}},
                    "l": [null, {"x": 10}],
                    "m": {"k": {"x": 7}},
                },
                {
                    "id": 2,
                    "s": {"a": 2, "inner": null},
                    "l": [],
                    "m": null,
                },
                {
                    "id": 3,
                    "s": null,
                    "l": null,
                    "m": {"k": null},
                },
            ]),
        );
        let expected = rows_to_batch(
            &table_schema,
            serde_json::json!([
                {
                    "id": 1,
                    "s": {"a": 1, "inner": {"y": null}},
                    "l": [null, {"x": null}],
                    "m": {"k": {"x": null}},
                },
                {
                    "id": 2,
                    "s": {"a": 2, "inner": null},
                    "l": [],
                    "m": null,
                },
                {
                    "id": 3,
                    "s": null,
                    "l": null,
                    "m": {"k": null},
                },
            ]),
        );
        let (_dir, path) = write_parquet(&file_batch);
        let out = read_parquet(&path, table_schema, vec![1, 2, 3, 4]).await;
        assert_eq!(out, vec![expected]);
    }
}
