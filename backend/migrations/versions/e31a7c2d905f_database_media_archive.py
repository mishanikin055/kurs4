"""Add durable PostgreSQL media storage without changing legacy completed images."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "e31a7c2d905f"
down_revision = "d20f4a6b819c"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "image_records",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("manifest", postgresql.JSONB(none_as_null=True), nullable=True),
        sa.CheckConstraint("width > 0 AND height > 0", name="image_dimensions"),
    )
    op.create_table(
        "image_artifacts",
        sa.Column(
            "image_id",
            sa.Uuid(),
            sa.ForeignKey("image_records.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("filename", sa.String(), primary_key=True),
        sa.Column("content_type", sa.String(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
    )
    op.create_table(
        "image_objects",
        sa.Column(
            "image_id",
            sa.Uuid(),
            sa.ForeignKey("image_records.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("object_id", sa.Uuid(), primary_key=True),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("bbox", postgresql.JSONB(), nullable=False),
        sa.Column("object_caption", sa.String(), nullable=True),
        sa.Column("attributes", postgresql.JSONB(), nullable=False),
    )
    op.create_index("image_objects_label_image_idx", "image_objects", ["label", "image_id"])
    op.create_index(
        "image_objects_attributes_idx", "image_objects", ["attributes"], postgresql_using="gin"
    )


def downgrade():
    op.drop_table("image_objects")
    op.drop_table("image_artifacts")
    op.drop_table("image_records")
