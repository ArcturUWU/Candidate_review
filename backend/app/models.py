import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Role(Base):
    __tablename__ = "roles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)

    scenarios: Mapped[list["Scenario"]] = relationship("Scenario", back_populates="role")


class RagCorpus(Base):
    __tablename__ = "rag_corpora"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    documents: Mapped[list["Document"]] = relationship("Document", back_populates="corpus", cascade="all, delete-orphan")
    scenarios: Mapped[list["Scenario"]] = relationship("Scenario", back_populates="rag_corpus")


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rag_corpus_id: Mapped[int] = mapped_column(ForeignKey("rag_corpora.id"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[Optional[dict[str, Any]]] = mapped_column("metadata", JSON, default={})

    corpus: Mapped["RagCorpus"] = relationship("RagCorpus", back_populates="documents")


class SqlScenario(Base):
    __tablename__ = "sql_scenarios"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    db_schema: Mapped[Optional[str]] = mapped_column(Text)
    reference_solutions: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    scenarios: Mapped[list["Scenario"]] = relationship("Scenario", back_populates="sql_scenario")


class Scenario(Base):
    __tablename__ = "scenarios"
    __table_args__ = (UniqueConstraint("role_id", "slug", name="uq_scenario_role_slug"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    difficulty: Mapped[Optional[str]] = mapped_column(String(50))
    tasks: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    rag_corpus_id: Mapped[Optional[int]] = mapped_column(ForeignKey("rag_corpora.id"))
    sql_scenario_id: Mapped[Optional[int]] = mapped_column(ForeignKey("sql_scenarios.id"))
    config: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)

    role: Mapped["Role"] = relationship("Role", back_populates="scenarios")
    rag_corpus: Mapped[Optional["RagCorpus"]] = relationship("RagCorpus", back_populates="scenarios")
    sql_scenario: Mapped[Optional["SqlScenario"]] = relationship("SqlScenario", back_populates="scenarios")
    sessions: Mapped[list["Session"]] = relationship("Session", back_populates="scenario")


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id: Mapped[Optional[str]] = mapped_column(String(128))
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id"), nullable=False)
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    state: Mapped[str] = mapped_column(String(50), default="active")
    current_task_id: Mapped[Optional[str]] = mapped_column(String(128))
    scores: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON)
    raw_log: Mapped[Optional[str]] = mapped_column(Text)

    scenario: Mapped["Scenario"] = relationship("Scenario", back_populates="sessions")
    role: Mapped["Role"] = relationship("Role")
    messages: Mapped[list["Message"]] = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    score_entries: Mapped[list["Score"]] = relationship("Score", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False, index=True)
    sender: Mapped[str] = mapped_column(String(50), nullable=False)  # candidate | model | system | tool
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    task_id: Mapped[Optional[str]] = mapped_column(String(128))

    session: Mapped["Session"] = relationship("Session", back_populates="messages")


class Score(Base):
    __tablename__ = "scores"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(String(128), nullable=False)
    points: Mapped[float] = mapped_column(Float, nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped["Session"] = relationship("Session", back_populates="score_entries")
