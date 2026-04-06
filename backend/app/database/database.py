from app.configuration import Configuration
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


config = Configuration.get_config()
postgres_config = config.postgres

DATABASE = {
    "drivername": "postgresql",
    "host": postgres_config.host,
    "port": postgres_config.port,
    "username": postgres_config.user,
    "password": postgres_config.password,
    "database": postgres_config.database,
}


engine = create_engine(URL.create(**DATABASE), pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_deals_table(engine):
    Base.metadata.create_all(bind=engine)


def init_database():
    # Ensure all ORM models are imported so SQLAlchemy registers their tables
    from app.database import analysis_job  # noqa: F401
    from app.database import history_status  # noqa: F401
    from app.database import user  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
