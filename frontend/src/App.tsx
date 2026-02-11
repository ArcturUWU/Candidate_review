import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_BASE =
  import.meta.env.VITE_API_URL ||
  (typeof window !== "undefined" && window.location.origin.includes("8000") ? "" : "http://127.0.0.1:8000");

type Task = {
  id: string;
  type: "theory" | "coding" | "sql";
  title: string;
  description?: string;
  description_for_candidate?: string;
  max_points?: number;
  tests_id?: string;
  sql_scenario_id?: string;
  language?: string;
  hints_allowed?: boolean;
};

type Role = {
  id: number;
  name: string;
  slug: string;
  description?: string;
};

type Scenario = {
  id: number;
  role_id: number;
  name: string;
  slug: string;
  description?: string;
  difficulty?: string;
  tasks?: Task[];
};

type Message = {
  sender: "candidate" | "model" | "system";
  text: string;
  created_at?: string;
  task_id?: string | null;
};

const defaultTasks: Task[] = [
  {
    id: "T1",
    type: "theory",
    title: "Основы регрессии",
    description: "Объяснить, что такое линейная регрессия, и перечислить метрики качества.",
    max_points: 5,
    hints_allowed: true,
  },
  {
    id: "C1",
    type: "coding",
    language: "python",
    title: "Логистическая регрессия",
    description_for_candidate: "Реализуйте логистическую регрессию без sklearn.",
    max_points: 10,
    tests_id: "logreg_basic",
  },
  {
    id: "SQL1",
    type: "sql",
    title: "Агрегация заказов",
    description_for_candidate: "По таблицам orders и customers посчитайте сумму заказов по городам.",
    max_points: 8,
    sql_scenario_id: "ecommerce_basic",
  },
];

async function fetchJson(path: string, init?: RequestInit) {
  const resp = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || "Request failed");
  }
  return resp.json();
}

function SectionHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="section-header">
      <h2>{title}</h2>
      {subtitle && <p className="muted">{subtitle}</p>}
    </div>
  );
}

type View = "landing" | "session" | "admin";

function App() {
  const [view, setView] = useState<View>("landing");
  const [roles, setRoles] = useState<Role[]>([]);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [selectedRole, setSelectedRole] = useState<number | null>(null);
  const [selectedScenario, setSelectedScenario] = useState<number | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingReply, setStreamingReply] = useState<string>("");
  const [streaming, setStreaming] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [codeDraft, setCodeDraft] = useState("# Напишите решение здесь\n");
  const [sqlDraft, setSqlDraft] = useState("select * from orders limit 5;");
  const [executionLog, setExecutionLog] = useState<string | null>(null);
  const [adminRoleDraft, setAdminRoleDraft] = useState({ name: "", slug: "", description: "" });
  const [adminScenarioDraft, setAdminScenarioDraft] = useState({
    role_id: "",
    name: "",
    slug: "",
    description: "",
    difficulty: "junior",
    tasks: JSON.stringify(defaultTasks, null, 2),
  });
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [currentTaskIndex, setCurrentTaskIndex] = useState(0);

  const selectedScenarioObj = useMemo(
    () => scenarios.find((s) => s.id === selectedScenario),
    [selectedScenario, scenarios],
  );

  const orderedTasks = selectedScenarioObj?.tasks || [];
  const currentTask = orderedTasks[currentTaskIndex] || null;

  useEffect(() => {
    const bootstrap = async () => {
      try {
        const [rolesResp, scenariosResp] = await Promise.all([
          fetchJson("/roles").catch(() => []),
          fetchJson("/scenarios").catch(() => []),
        ]);
        setRoles(rolesResp.length ? rolesResp : sampleRoles());
        setScenarios(scenariosResp.length ? scenariosResp : sampleScenarios());
      } catch (err) {
        console.error(err);
        setRoles(sampleRoles());
        setScenarios(sampleScenarios());
      }
    };
    bootstrap();
  }, []);

  useEffect(() => {
    if (selectedRole && !selectedScenario) {
      const first = scenarios.find((s) => s.role_id === selectedRole);
      if (first) setSelectedScenario(first.id);
    }
  }, [selectedRole, selectedScenario, scenarios]);

  const sampleRoles = (): Role[] => [
    { id: 1, name: "Data Scientist", slug: "ds", description: "ML, эксперименты, метрики" },
    { id: 2, name: "Backend", slug: "backend", description: "API, очереди, устойчивость" },
    { id: 3, name: "Data Engineer", slug: "de", description: "ETL, SQL, пайплайны" },
  ];

  const sampleScenarios = (): Scenario[] => [
    {
      id: 101,
      role_id: 1,
      name: "DS — Junior ML",
      slug: "ds-junior-ml",
      description: "Регрессия, классификация, SQL основы",
      difficulty: "junior",
      tasks: defaultTasks,
    },
    {
      id: 102,
      role_id: 1,
      name: "DS — Product ML",
      slug: "ds-product-ml",
      description: "A/B, метрики продукта, рекомендации",
      difficulty: "middle",
      tasks: [
        { id: "T-metrics", type: "theory", title: "Метрики A/B", max_points: 5, hints_allowed: true },
        {
          id: "SQL-ab",
          type: "sql",
          title: "Конверсия по когорте",
          description_for_candidate: "Напишите запрос конверсии по дню регистрации.",
          sql_scenario_id: "ab_product",
          max_points: 8,
        },
      ],
    },
    {
      id: 201,
      role_id: 2,
      name: "Backend — REST",
      slug: "be-rest",
      description: "API дизайн, идемпотентность, очереди",
      difficulty: "middle",
      tasks: [
        { id: "T-REST", type: "theory", title: "PUT vs PATCH идемпотентность", max_points: 5 },
        {
          id: "C-BE",
          type: "coding",
          language: "python",
          title: "Очередь задач",
          description_for_candidate: "Реализуйте очередь с ack/nack.",
          tests_id: "queue_basic",
          max_points: 8,
        },
      ],
    },
    {
      id: 202,
      role_id: 2,
      name: "Backend — Resilience",
      slug: "be-resilience",
      description: "Ретраи, троттлинг, circuit breaker",
      difficulty: "senior",
      tasks: [
        { id: "T-circuit", type: "theory", title: "Circuit breaker", max_points: 6, hints_allowed: true },
        {
          id: "C-rate",
          type: "coding",
          language: "python",
          title: "Rate limiter",
          description_for_candidate: "Сделайте токен-бакет.",
          tests_id: "rate_limiter",
          max_points: 9,
        },
      ],
    },
    {
      id: 301,
      role_id: 3,
      name: "DE — Pipelines",
      slug: "de-pipelines",
      description: "Инкрементальные пайплайны, буферизация, SLA",
      difficulty: "middle",
      tasks: [
        { id: "T-de-incr", type: "theory", title: "Инкрементальные загрузки", max_points: 5, hints_allowed: true },
        {
          id: "SQL-de-agg",
          type: "sql",
          title: "Агрегация событий",
          description_for_candidate: "Посчитайте DAU по регионам из таблицы events.",
          sql_scenario_id: "events_basic",
          max_points: 8,
        },
      ],
    },
    {
      id: 302,
      role_id: 3,
      name: "DE — Warehousing",
      slug: "de-warehousing",
      description: "Моделирование данных, SCD, оркестрация",
      difficulty: "senior",
      tasks: [
        { id: "T-scd", type: "theory", title: "SCD типы", max_points: 6 },
        {
          id: "SQL-scd",
          type: "sql",
          title: "SCD Type 2 обновление",
          description_for_candidate: "Напишите SQL, который добавляет новую версию записи клиента.",
          sql_scenario_id: "scd_customers",
          max_points: 9,
        },
      ],
    },
  ];

  const pushMessage = (msg: Message) => setMessages((prev) => [...prev, msg]);

  const streamModel = async () => {
    if (!sessionId) return;
    setStreaming(true);
    setStreamingReply("");
    let accumulated = "";
    try {
      const resp = await fetch(`${API_BASE}/sessions/${sessionId}/lm/chat-stream`, { method: "GET" });
      if (!resp.body) throw new Error("Нет body у ответа");
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          if (!part.trim().startsWith("data:")) continue;
          const raw = part.trim().slice(5).trim();
          if (!raw) continue;
          const data = JSON.parse(raw);
          if (data.type === "token") {
            accumulated += data.content;
            setStreamingReply((prev) => prev + data.content);
          }
          if (data.type === "error") {
            pushMessage({ sender: "system", text: `Ошибка модели: ${data.detail || data.content}` });
          }
          if (data.type === "done") {
            if (data.content) {
              accumulated = data.content;
              setStreamingReply(data.content);
            }
          }
        }
      }
    } catch (err: any) {
      pushMessage({ sender: "system", text: `Стрим не удался: ${err.message}` });
    } finally {
      setStreaming(false);
      if (accumulated) {
        pushMessage({ sender: "model", text: accumulated });
        setStreamingReply("");
      }
    }
  };

  const startSession = async () => {
    if (!selectedRole || !selectedScenario) {
      setStatus("Выберите роль и сценарий");
      return;
    }
    setLoading(true);
    try {
      const resp = await fetchJson("/sessions", {
        method: "POST",
        body: JSON.stringify({
          role_id: selectedRole,
          scenario_id: selectedScenario,
        }),
      });
      setSessionId(resp.id);
      setMessages([
        {
          sender: "system",
          text: "Сессия создана. Модель сейчас представится и обозначит план.",
        },
      ]);
      setStatus("Сессия активна");
      setView("session");
      setCurrentTaskIndex(0);
      await streamModel();
    } catch (err: any) {
      setStatus(err.message);
    } finally {
      setLoading(false);
    }
  };

  const sendChatMessage = async () => {
    if (!chatInput.trim() || !sessionId) return;
    const outgoing: Message = { sender: "candidate", text: chatInput, task_id: currentTask?.id };
    pushMessage(outgoing);
    setChatInput("");
    try {
      await fetchJson(`/sessions/${sessionId}/messages`, {
        method: "POST",
        body: JSON.stringify(outgoing),
      });
      await streamModel();
    } catch (err: any) {
      pushMessage({ sender: "system", text: `Модель недоступна: ${err.message}` });
    }
  };

  const submitCode = async (task?: Task) => {
    if (!sessionId || !task) {
      setExecutionLog("Нет активной сессии или кода");
      return;
    }
    try {
      const resp = await fetchJson(`/sessions/${sessionId}/tasks/${task.id}/submit_code`, {
        method: "POST",
        body: JSON.stringify({
          code: codeDraft,
          language: task.language || "python",
          tests_id: task.tests_id || "sample_tests",
        }),
      });
      setExecutionLog(JSON.stringify(resp, null, 2));
    } catch (err: any) {
      setExecutionLog(`Ошибка отправки: ${err.message}`);
    }
  };

  const submitSql = async (task?: Task) => {
    if (!sessionId || !task) {
      setExecutionLog("Нет активной сессии или SQL");
      return;
    }
    try {
      const resp = await fetchJson(`/sessions/${sessionId}/tasks/${task.id}/submit_sql`, {
        method: "POST",
        body: JSON.stringify({
          query: sqlDraft,
          sql_scenario_id: task.sql_scenario_id || "demo_sql",
        }),
      });
      setExecutionLog(JSON.stringify(resp, null, 2));
    } catch (err: any) {
      setExecutionLog(`Ошибка SQL: ${err.message}`);
    }
  };

  const submitRole = async () => {
    try {
      const resp = await fetchJson("/roles", {
        method: "POST",
        body: JSON.stringify(adminRoleDraft),
      });
      setRoles((prev) => [...prev, resp]);
      setAdminRoleDraft({ name: "", slug: "", description: "" });
      setStatus("Роль сохранена");
    } catch (err: any) {
      setStatus(err.message);
    }
  };

  const submitScenario = async () => {
    try {
      const parsedTasks = JSON.parse(adminScenarioDraft.tasks);
      const payload = {
        role_id: Number(adminScenarioDraft.role_id),
        name: adminScenarioDraft.name,
        slug: adminScenarioDraft.slug,
        description: adminScenarioDraft.description,
        difficulty: adminScenarioDraft.difficulty,
        tasks: parsedTasks,
      };
      const resp = await fetchJson("/scenarios", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setScenarios((prev) => [...prev, resp]);
      setStatus("Сценарий сохранен");
    } catch (err: any) {
      setStatus(`Ошибка сценария: ${err.message}`);
    }
  };

  const goNextTask = () => {
    if (currentTaskIndex < orderedTasks.length - 1) {
      setCurrentTaskIndex((i) => i + 1);
    }
  };

  const goPrevTask = () => {
    if (currentTaskIndex > 0) {
      setCurrentTaskIndex((i) => i - 1);
    }
  };

  const taskHint = (task: Task | null) => {
    if (!task) return "";
    if (task.type === "theory") return "Ответьте в чате. Если задание допускает подсказки, модель предложит наводку.";
    if (task.type === "coding") return "Напишите код ниже и отправьте. После submit редактор блокируется в реальном UI.";
    if (task.type === "sql") return "Напишите SQL ниже и отправьте. После submit редактор блокируется в реальном UI.";
    return "";
  };

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Chat-Review</p>
          <h1>
            Собеседование в чате с кодом и SQL <span className="accent">под контролем</span>
          </h1>
          <p className="muted">
            Выберите роль и сценарий, создайте сессию, а дальше — структурированный диалог с моделью-интервьюером.
            Песочницы для кода и SQL подключены, подсказки и оценки идут через tools.
          </p>
          <div className="hero-actions">
            {view !== "admin" && (
              <button className="primary" disabled={loading} onClick={startSession}>
                {sessionId ? "Начать" : "Создать сессию и начать"}
              </button>
            )}
            <button className="ghost" onClick={() => setView(view === "admin" ? "landing" : "admin")}>
              {view === "admin" ? "Вернуться" : "Открыть админку"}
            </button>
            {status && <span className="pill">{status}</span>}
          </div>
        </div>
        <div className="status-card">
          <p className="muted">Сессия</p>
          <h3>{sessionId || "не создана"}</h3>
          <p className="muted">{selectedScenarioObj?.name || "Выберите сценарий"}</p>
        </div>
      </header>

      {view === "landing" && (
        <>
          <div className="grid two-column">
            <section className="panel">
              <SectionHeader title="1. Роль" subtitle="Роли задают контекст промпта" />
              <div className="cards">
                {roles.map((role) => (
                  <button
                    key={role.id}
                    className={`card ${selectedRole === role.id ? "active" : ""}`}
                    onClick={() => {
                      setSelectedRole(role.id);
                      setSelectedScenario(null);
                    }}
                  >
                    <div className="card-title">
                      <span>{role.name}</span>
                      <span className="muted">{role.slug}</span>
                    </div>
                    <p className="muted">{role.description}</p>
                  </button>
                ))}
              </div>
            </section>

            <section className="panel">
              <SectionHeader title="2. Сценарий" subtitle="Стандартизированные задачи" />
              <div className="scenario-list">
                {(scenarios || [])
                  .filter((s) => !selectedRole || s.role_id === selectedRole)
                  .map((scenario) => (
                    <div
                      key={scenario.id}
                      className={`scenario ${selectedScenario === scenario.id ? "active" : ""}`}
                      onClick={() => setSelectedScenario(scenario.id)}
                    >
                      <div className="scenario-head">
                        <div>
                          <p className="label">{scenario.difficulty}</p>
                          <h4>{scenario.name}</h4>
                        </div>
                        <p className="muted">{scenario.slug}</p>
                      </div>
                      <p className="muted">{scenario.description}</p>
                      <div className="chips">
                        {(scenario.tasks || []).map((task) => (
                          <span key={task.id} className="chip">
                            {task.type.toUpperCase()} · {task.id}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
              </div>
            </section>
          </div>
          <div className="center">
            <button className="primary" onClick={startSession} disabled={loading}>
              Начать интервью
            </button>
          </div>
        </>
      )}

      {view === "session" && (
        <>
          <section className="panel">
            <SectionHeader title="Чат" subtitle="Приветствие сверху, затем решения заданий" />
              <div className="chat">
                <div className="messages">
                  {messages.map((msg, idx) => (
                    <div key={idx} className={`message ${msg.sender}`}>
                      <div className="message-meta">
                        <span>{msg.sender}</span>
                        {msg.task_id && <span className="pill small">task {msg.task_id}</span>}
                      </div>
                      <ReactMarkdown>{msg.text}</ReactMarkdown>
                    </div>
                  ))}
                  {streaming && (
                    <div className="message model">
                      <div className="message-meta">
                        <span>model • streaming</span>
                      </div>
                      <ReactMarkdown>{streamingReply || "…"}</ReactMarkdown>
                    </div>
                  )}
                </div>
              <div className="composer">
                <textarea
                  placeholder="Ваш ответ или вопрос..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                />
                <button className="primary" onClick={sendChatMessage} disabled={!sessionId || streaming}>
                  Отправить
                </button>
              </div>
            </div>
          </section>

          <section className="panel">
            <SectionHeader title="Задания" subtitle="Показываем по одному, сохраняем контекст" />
            <div className="task-nav">
              <div>
                <p className="label">Текущее задание</p>
                <h4>
                  {currentTaskIndex + 1}/{orderedTasks.length} · {currentTask?.id || "—"} · {currentTask?.title || "Нет"}
                </h4>
                <p className="muted">{taskHint(currentTask)}</p>
              </div>
              <div className="task-nav-buttons">
                <button className="ghost" onClick={goPrevTask} disabled={currentTaskIndex === 0}>
                  Назад
                </button>
                <button className="ghost" onClick={goNextTask} disabled={currentTaskIndex >= orderedTasks.length - 1}>
                  Следующее
                </button>
              </div>
            </div>

            <div className="task-details">
              <h4>{selectedScenarioObj?.name || "Задачи появятся после выбора"}</h4>
              <p className="muted">
                {(selectedScenarioObj?.tasks || []).length
                  ? `${(selectedScenarioObj?.tasks || []).length} заданий в сценарии`
                  : "Добавьте задачи в админке"}
              </p>

              {currentTask?.type === "theory" && (
                <div className="editor">
                  <div className="editor-head">
                    <div>
                      <p className="label">Theory</p>
                      <h5>{currentTask.title}</h5>
                      <p className="muted">{currentTask.description}</p>
                    </div>
                  </div>
                  <p className="muted">Ответ дайте в чате выше. Модель задаст уточнения и выставит баллы.</p>
                </div>
              )}

              {currentTask?.type === "coding" && (
                <div className="editor">
                  <div className="editor-head">
                    <div>
                      <p className="label">Coding</p>
                      <h5>{currentTask.title}</h5>
                      <p className="muted">
                        {currentTask.description_for_candidate || currentTask.description || "Напишите решение ниже"}
                      </p>
                    </div>
                    <button className="ghost" onClick={() => submitCode(currentTask)} disabled={!sessionId}>
                      Submit code
                    </button>
                  </div>
                  <textarea value={codeDraft} onChange={(e) => setCodeDraft(e.target.value)} className="code tall" />
                </div>
              )}

              {currentTask?.type === "sql" && (
                <div className="editor">
                  <div className="editor-head">
                    <div>
                      <p className="label">SQL</p>
                      <h5>{currentTask.title}</h5>
                      <p className="muted">
                        {currentTask.description_for_candidate || currentTask.description || "Напишите SQL-запрос"}
                      </p>
                    </div>
                    <button className="ghost" onClick={() => submitSql(currentTask)} disabled={!sessionId}>
                      Submit SQL
                    </button>
                  </div>
                  <textarea value={sqlDraft} onChange={(e) => setSqlDraft(e.target.value)} className="code tall" />
                </div>
              )}

              {executionLog && (
                <div className="log">
                  <p className="label">Результат песочницы</p>
                  <pre>{executionLog}</pre>
                </div>
              )}
            </div>
          </section>
        </>
      )}

      {view === "admin" && (
        <section className="panel">
          <SectionHeader title="Админка" subtitle="Роли, сценарии, документы" />
          <div className="grid two-column">
            <div className="form">
              <h4>Новая роль</h4>
              <input
                placeholder="Название"
                value={adminRoleDraft.name}
                onChange={(e) => setAdminRoleDraft({ ...adminRoleDraft, name: e.target.value })}
              />
              <input
                placeholder="Slug"
                value={adminRoleDraft.slug}
                onChange={(e) => setAdminRoleDraft({ ...adminRoleDraft, slug: e.target.value })}
              />
              <textarea
                placeholder="Описание"
                value={adminRoleDraft.description}
                onChange={(e) => setAdminRoleDraft({ ...adminRoleDraft, description: e.target.value })}
              />
              <button onClick={submitRole}>Сохранить роль</button>
            </div>

            <div className="form">
              <h4>Новый сценарий</h4>
              <input
                placeholder="Role ID"
                value={adminScenarioDraft.role_id}
                onChange={(e) => setAdminScenarioDraft({ ...adminScenarioDraft, role_id: e.target.value })}
              />
              <input
                placeholder="Название"
                value={adminScenarioDraft.name}
                onChange={(e) => setAdminScenarioDraft({ ...adminScenarioDraft, name: e.target.value })}
              />
              <input
                placeholder="Slug"
                value={adminScenarioDraft.slug}
                onChange={(e) => setAdminScenarioDraft({ ...adminScenarioDraft, slug: e.target.value })}
              />
              <textarea
                placeholder="Описание"
                value={adminScenarioDraft.description}
                onChange={(e) => setAdminScenarioDraft({ ...adminScenarioDraft, description: e.target.value })}
              />
              <textarea
                placeholder="tasks в стандартизированном формате"
                value={adminScenarioDraft.tasks}
                onChange={(e) => setAdminScenarioDraft({ ...adminScenarioDraft, tasks: e.target.value })}
                className="code"
              />
              <button onClick={submitScenario}>Сохранить сценарий</button>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

export default App;
