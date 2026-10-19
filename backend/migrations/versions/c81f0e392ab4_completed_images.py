"""Add completed images without modifying legacy jobs or results."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c81f0e392ab4"
down_revision = "b7d91f4e20ac"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "processed_images",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("original_path", sa.String(), nullable=False),
        sa.Column("normalized_path", sa.String(), nullable=False),
        sa.Column("result_path", sa.String(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("models", postgresql.JSONB(), nullable=False),
        sa.Column("detection", postgresql.JSONB(), nullable=False),
        sa.Column("classification", postgresql.JSONB(), nullable=False),
        sa.Column("annotation", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint("width > 0 AND height > 0", name="processed_dimensions"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade():
    raise RuntimeError("Completed results removal is irreversible; back up the table first")
