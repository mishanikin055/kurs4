"""Remove the retired jobs schema after an explicit backup and reset request."""

from alembic import op

revision = "d20f4a6b819c"
down_revision = "c81f0e392ab4"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table("model_results")
    op.drop_table("jobs")
    op.drop_table("images")


def downgrade():
    raise RuntimeError("Legacy data removal is irreversible; restore a database backup")
