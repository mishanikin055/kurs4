"""Keep image references and compact stage results.

Removed metadata and alternative predictions cannot be reconstructed on downgrade.
"""

from alembic import op

revision = "b7d91f4e20ac"
down_revision = "a2f004752f43"
branch_labels = None
depends_on = None


def upgrade():
    # The API's five-second query limit is too short for rewriting existing JSONB rows.
    op.execute("SET LOCAL statement_timeout = 0")
    # Preserve array order: API classification labels follow detection order.
    op.execute("""
        UPDATE model_results SET payload = jsonb_build_object('objects', COALESCE((
            SELECT jsonb_agg(jsonb_build_object(
                'object_id', obj->'object_id', 'bbox', obj->'bbox', 'label', obj->'label'
            ) ORDER BY position)
            FROM jsonb_array_elements(payload->'objects') WITH ORDINALITY AS items(obj, position)
        ), '[]'::jsonb))
        WHERE stage = 'detection' AND payload IS NOT NULL
    """)
    op.execute("""
        UPDATE model_results SET payload = jsonb_build_object('objects', COALESCE((
            SELECT jsonb_agg(
                jsonb_build_object('object_id', obj->'object_id') ||
                CASE WHEN obj->>'status' = 'succeeded' THEN
                    jsonb_build_object('label', (
                        SELECT prediction->'label'
                        FROM jsonb_array_elements(obj->'top_k') AS prediction
                        ORDER BY (prediction->>'score')::double precision DESC
                        LIMIT 1
                    ))
                ELSE jsonb_build_object('error', obj->'error') END
                ORDER BY position
            )
            FROM jsonb_array_elements(payload->'objects') WITH ORDINALITY AS items(obj, position)
        ), '[]'::jsonb))
        WHERE stage = 'classification' AND payload IS NOT NULL
    """)
    op.drop_constraint("image_size", "images", type_="check")
    for column in (
        "original_filename",
        "original_sha256",
        "original_mime",
        "original_size_bytes",
        "normalized_sha256",
        "normalization_version",
        "created_at",
    ):
        op.drop_column("images", column)
    for column in ("published_at", "publish_error"):
        op.drop_column("jobs", column)
    op.drop_constraint("result_duration", "model_results", type_="check")
    op.drop_constraint("uq_model_results_job_stage", "model_results", type_="unique")
    op.drop_constraint("model_results_pkey", "model_results", type_="primary")
    for column in (
        "id",
        "weights_sha256",
        "parameters",
        "schema_version",
        "started_at",
        "finished_at",
        "duration_ms",
    ):
        op.drop_column("model_results", column)
    op.create_primary_key("model_results_pkey", "model_results", ["job_id", "stage"])


def downgrade():
    raise RuntimeError("Compact results are irreversible; restore a pre-migration database backup")
