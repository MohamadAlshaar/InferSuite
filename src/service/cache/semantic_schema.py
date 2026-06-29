from __future__ import annotations

from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility


SEMANTIC_CACHE_PRIMARY_KEY_FIELD = "pk"
SEMANTIC_CACHE_ACCEPT_KEY_FIELD = "accept_key"
SEMANTIC_CACHE_VECTOR_FIELD = "embedding"

SEMANTIC_CACHE_REQUIRED_FIELDS = (
    SEMANTIC_CACHE_PRIMARY_KEY_FIELD,
    SEMANTIC_CACHE_ACCEPT_KEY_FIELD,
    SEMANTIC_CACHE_VECTOR_FIELD,
)


def get_semantic_cache_field_names() -> list[str]:
    return list(SEMANTIC_CACHE_REQUIRED_FIELDS)


def build_semantic_cache_fields(vector_dim: int) -> list[FieldSchema]:
    return [
        FieldSchema(
            name=SEMANTIC_CACHE_PRIMARY_KEY_FIELD,
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=128,
        ),
        FieldSchema(
            name=SEMANTIC_CACHE_ACCEPT_KEY_FIELD,
            dtype=DataType.VARCHAR,
            max_length=512,
        ),
        FieldSchema(
            name=SEMANTIC_CACHE_VECTOR_FIELD,
            dtype=DataType.FLOAT_VECTOR,
            dim=int(vector_dim),
        ),
    ]


def build_semantic_cache_schema(vector_dim: int) -> CollectionSchema:
    return CollectionSchema(
        fields=build_semantic_cache_fields(vector_dim),
        description="Semantic cache vectors for llm-service-kernel",
    )


def get_semantic_cache_index_params() -> dict[str, Any]:
    return {
        "index_type": "AUTOINDEX",
        "metric_type": "COSINE",
        "params": {},
    }


def get_semantic_cache_output_fields() -> list[str]:
    return [
        SEMANTIC_CACHE_PRIMARY_KEY_FIELD,
        SEMANTIC_CACHE_ACCEPT_KEY_FIELD,
    ]


def get_semantic_cache_insert_payload(
    *,
    entry_id: str,
    accept_key: str,
    vector: list[float],
) -> list[list[Any]]:
    return [
        [str(entry_id)],
        [str(accept_key)],
        [list(vector)],
    ]


def validate_semantic_cache_collection(
    collection: Collection,
    *,
    expected_vector_dim: int | None = None,
) -> None:
    schema = getattr(collection, "schema", None)
    if schema is None:
        raise RuntimeError("Milvus collection has no schema")

    fields = {field.name: field for field in schema.fields}
    missing = [name for name in SEMANTIC_CACHE_REQUIRED_FIELDS if name not in fields]
    if missing:
        raise RuntimeError(
            f"Milvus collection '{collection.name}' is missing required semantic-cache fields: {missing}"
        )

    pk_field = fields[SEMANTIC_CACHE_PRIMARY_KEY_FIELD]
    accept_key_field = fields[SEMANTIC_CACHE_ACCEPT_KEY_FIELD]
    vector_field = fields[SEMANTIC_CACHE_VECTOR_FIELD]

    if pk_field.dtype != DataType.VARCHAR:
        raise RuntimeError(
            f"Milvus collection '{collection.name}' field '{SEMANTIC_CACHE_PRIMARY_KEY_FIELD}' "
            f"must be VARCHAR, got {pk_field.dtype}"
        )

    if accept_key_field.dtype != DataType.VARCHAR:
        raise RuntimeError(
            f"Milvus collection '{collection.name}' field '{SEMANTIC_CACHE_ACCEPT_KEY_FIELD}' "
            f"must be VARCHAR, got {accept_key_field.dtype}"
        )

    if vector_field.dtype != DataType.FLOAT_VECTOR:
        raise RuntimeError(
            f"Milvus collection '{collection.name}' field '{SEMANTIC_CACHE_VECTOR_FIELD}' "
            f"must be FLOAT_VECTOR, got {vector_field.dtype}"
        )

    if expected_vector_dim is not None:
        params = getattr(vector_field, "params", {}) or {}
        actual_dim = params.get("dim")
        if actual_dim is not None and int(actual_dim) != int(expected_vector_dim):
            raise RuntimeError(
                f"Milvus collection '{collection.name}' field '{SEMANTIC_CACHE_VECTOR_FIELD}' "
                f"has dim={actual_dim}, expected {expected_vector_dim}"
            )


_MILVUS_RPC_TIMEOUT = 30


def open_semantic_cache_collection(
    *,
    alias: str,
    collection_name: str,
    expected_vector_dim: int | None = None,
) -> Collection:
    if not utility.has_collection(collection_name, using=alias, timeout=_MILVUS_RPC_TIMEOUT):
        raise RuntimeError(
            f"Semantic cache Milvus collection '{collection_name}' does not exist"
        )

    collection = Collection(name=collection_name, using=alias)
    validate_semantic_cache_collection(
        collection,
        expected_vector_dim=expected_vector_dim,
    )
    return collection


def create_semantic_cache_collection(
    *,
    alias: str,
    collection_name: str,
    vector_dim: int,
) -> Collection:
    if utility.has_collection(collection_name, using=alias, timeout=_MILVUS_RPC_TIMEOUT):
        return open_semantic_cache_collection(
            alias=alias,
            collection_name=collection_name,
            expected_vector_dim=vector_dim,
        )

    collection = Collection(
        name=collection_name,
        schema=build_semantic_cache_schema(vector_dim),
        using=alias,
    )
    collection.create_index(
        field_name=SEMANTIC_CACHE_VECTOR_FIELD,
        index_params=get_semantic_cache_index_params(),
        timeout=_MILVUS_RPC_TIMEOUT,
    )
    return collection


def ensure_semantic_cache_collection(
    *,
    alias: str,
    collection_name: str,
    vector_dim: int,
    create_if_missing: bool,
) -> Collection:
    if utility.has_collection(collection_name, using=alias, timeout=_MILVUS_RPC_TIMEOUT):
        return open_semantic_cache_collection(
            alias=alias,
            collection_name=collection_name,
            expected_vector_dim=vector_dim,
        )

    if not create_if_missing:
        raise RuntimeError(
            f"Semantic cache Milvus collection '{collection_name}' does not exist"
        )

    return create_semantic_cache_collection(
        alias=alias,
        collection_name=collection_name,
        vector_dim=vector_dim,
    )
