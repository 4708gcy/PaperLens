"""SQLAlchemy ORM 模型：论文 / 对话 / 消息"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Paper(Base):
    """论文表"""
    __tablename__ = "papers"

    paper_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    authors = Column(String(500), default="")
    file_path = Column(String(500), nullable=False)
    markdown_path = Column(String(500), default="")
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    status = Column(String(20), default="pending")  # pending/processing/indexed/failed
    abstract = Column(Text, default="")
    create_dt = Column(DateTime, default=datetime.utcnow)
    update_dt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Conversation(Base):
    """对话表"""
    __tablename__ = "conversations"

    conversation_id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(100), nullable=False, index=True)
    title = Column(String(200), default="未命名对话")
    paper_ids = Column(String(200), default="")
    create_dt = Column(DateTime, default=datetime.utcnow)


class Message(Base):
    """消息表"""
    __tablename__ = "messages"

    message_id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.conversation_id"), index=True)
    role = Column(String(20), nullable=False)  # user / assistant
    content = Column(Text, default="")
    intent = Column(String(20), default="")  # qa / analyze / synthesize
    create_dt = Column(DateTime, default=datetime.utcnow)


class Database:
    """数据库连接管理（单例）"""
    _engine = None
    _SessionLocal = None

    @classmethod
    def init(cls, db_path: str = "data/paperlens.db"):
        """初始化数据库，创建所有表"""
        import os
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        cls._engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False}
        )
        cls._SessionLocal = sessionmaker(bind=cls._engine, expire_on_commit=False)
        Base.metadata.create_all(cls._engine)

    @classmethod
    def get_session(cls):
        """获取一个 session（用完要 close）"""
        if cls._SessionLocal is None:
            raise RuntimeError("Database 未初始化，请先调用 Database.init()")
        return cls._SessionLocal()
