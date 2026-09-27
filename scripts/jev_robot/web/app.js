const dom = {
  form: document.querySelector("#goal-form"),
  input: document.querySelector("#goal-input"),
  run: document.querySelector("#run-button"),
  error: document.querySelector("#form-error"),
  statusDot: document.querySelector("#status-dot"),
  connection: document.querySelector("#connection-label"),
  contextEmpty: document.querySelector("#context-empty"),
  context: document.querySelector("#context-content"),
  sceneState: document.querySelector("#scene-state"),
  decisionEmpty: document.querySelector("#decision-empty"),
  decisionFeed: document.querySelector("#decision-feed"),
  decisionState: document.querySelector("#decision-state"),
  telemetryEmpty: document.querySelector("#telemetry-empty"),
  telemetryFeed: document.querySelector("#telemetry-feed"),
  robotState: document.querySelector("#robot-state"),
  cameraDot: document.querySelector("#camera-dot"),
  cameraLabel: document.querySelector("#camera-label"),
  cameraFrame: document.querySelector("#wrist-frame"),
  cameraOffline: document.querySelector("#camera-offline"),
  cameraSequence: document.querySelector("#camera-sequence"),
  cameraResolution: document.querySelector("#camera-resolution"),
  cameraSource: document.querySelector("#camera-source"),
  cameraMessage: document.querySelector("#camera-message"),
  cameraLiveBadge: document.querySelector("#camera-live-badge"),
};

const app = { cursor: 0, turns: new Map(), requestInFlight: false, cameraSequence: -1 };

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function pretty(value) {
  return String(value ?? "unknown").replaceAll("_", " ");
}

function percent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
}

function setConnection(label, mode = "ready") {
  dom.connection.textContent = label;
  dom.statusDot.className = `status-dot ${mode === "ready" ? "" : mode}`.trim();
}

function resetWorkspace() {
  app.cursor = 0;
  app.turns.clear();
  dom.context.textContent = "";
  dom.decisionFeed.textContent = "";
  dom.telemetryFeed.textContent = "";
  dom.contextEmpty.classList.remove("hidden");
  dom.decisionEmpty.classList.remove("hidden");
  dom.telemetryEmpty.classList.remove("hidden");
  dom.context.classList.add("hidden");
  dom.sceneState.textContent = "Starting";
  dom.decisionState.textContent = "Preparing question";
  dom.robotState.textContent = "Connecting";
}

function section(label) {
  const wrapper = node("section", "context-section");
  wrapper.append(node("div", "section-label", label));
  return wrapper;
}

function positionText(item) {
  const pose = item.pose || item.latest_pose || {};
  const values = pose.position_m;
  if (!Array.isArray(values) || values.length !== 3) return "Position unavailable";
  return values.map((value) => `${(Number(value) * 1000).toFixed(0)}`).join(", ") + " mm";
}

function objectCard(item, remembered = false) {
  const card = node("div", "object-card");
  const head = node("div", "object-card-head");
  head.append(
    node("span", "object-name", item.description || item.label || "Unknown object"),
    node("span", "object-id", item.object_id || item.id || "untracked"),
  );
  card.append(head);
  const confidence = item.confidence ?? item.latest_confidence;
  const details = [positionText(item)];
  if (confidence !== undefined) details.push(`${percent(confidence)} confidence`);
  if (remembered) {
    details.push(item.currently_visible ? "visible now" : "remembered");
    if (item.stale) details.push("stale");
  }
  card.append(node("div", "object-meta", details.join(" · ")));
  return card;
}

function renderContext(payload) {
  const state = payload.state || {};
  const scene = state.current_scene || {};
  const memory = state.scene_memory || {};
  dom.context.textContent = "";
  dom.contextEmpty.classList.add("hidden");
  dom.context.classList.remove("hidden");
  dom.sceneState.textContent = `Turn ${payload.turn} · frame ${scene.sequence ?? "—"}`;

  const goal = section("Active goal");
  goal.append(node("p", "goal-quote", state.user_goal || "—"));
  dom.context.append(goal);

  const visibleSection = section(`Current wrist view · ${(scene.visible_objects || []).length} objects`);
  const visibleList = node("div", "object-list");
  const visible = scene.visible_objects || [];
  if (visible.length) visible.forEach((item) => visibleList.append(objectCard(item)));
  else visibleList.append(node("div", "object-meta", "No object accepted in this frame."));
  visibleSection.append(visibleList);
  dom.context.append(visibleSection);

  const memorySection = section(`Scene memory · revision ${memory.revision ?? 0}`);
  const memoryList = node("div", "object-list");
  const known = memory.known_objects || [];
  if (known.length) known.forEach((item) => memoryList.append(objectCard(item, true)));
  else memoryList.append(node("div", "object-meta", "No remembered objects yet."));
  memorySection.append(memoryList);
  dom.context.append(memorySection);

  const unknownSection = section("Unknown / unobserved");
  const tags = node("div", "unknown-list");
  const unknown = scene.unknown_regions || memory.unknown_regions || [];
  (unknown.length ? unknown : ["none"]).forEach((value) => tags.append(node("span", "tag", pretty(value))));
  unknownSection.append(tags);
  dom.context.append(unknownSection);

  const recent = state.recent_action_results || [];
  if (recent.length) {
    const recentSection = section("Recent results");
    const list = node("div", "object-list");
    recent.slice(-4).forEach((result) => {
      const card = objectCard({
        label: `${result.success ? "✓" : "×"} ${pretty(result.action)}`,
        object_id: result.success ? "SUCCESS" : "FAILED",
      });
      card.querySelector(".object-meta").textContent = result.message || "No details";
      list.append(card);
    });
    recentSection.append(list);
    dom.context.append(recentSection);
  }
}

function createTurn(payload) {
  dom.decisionEmpty.classList.add("hidden");
  const card = node("section", "turn-card");
  card.dataset.turn = String(payload.turn);
  const head = node("div", "turn-head");
  head.append(
    node("span", "turn-number", `TURN ${String(payload.turn).padStart(2, "0")}`),
    node("span", "turn-phase", "Question"),
  );
  const question = node("div", "question");
  question.append(
    node("div", "question-label", "JEV question"),
    node("h3", "", payload.question || "What should the robot do next?"),
  );
  const choices = node("div", "choice-list");
  (payload.choices || []).forEach((choice) => {
    const item = node("div", "choice-item");
    item.dataset.choice = choice.name;
    item.append(
      node("div", "choice-name", pretty(choice.name)),
      node("div", "choice-description", choice.description || ""),
    );
    choices.append(item);
  });
  const waiting = node("div", "waiting");
  waiting.append(node("span", "spinner"), node("span", "", "Waiting for JEV"));
  card.append(head, question, choices, waiting);
  dom.decisionFeed.prepend(card);
  app.turns.set(Number(payload.turn), card);
  dom.decisionState.textContent = `Turn ${payload.turn} · waiting for JEV`;
  return card;
}

function renderChoice(payload) {
  const turn = Number(payload.turn);
  const card = app.turns.get(turn) || createTurn({
    turn,
    question: "What should the robot do next?",
    choices: Object.keys(payload.probabilities || {}).map((name) => ({ name, description: "" })),
  });
  card.querySelector(".turn-phase").textContent = "Answered";
  card.querySelector(".waiting")?.remove();
  const probabilities = payload.probabilities || {};
  card.querySelectorAll(".choice-item").forEach((item) => {
    const name = item.dataset.choice;
    if (name === payload.next_action) item.classList.add("selected");
    if (probabilities[name] !== undefined) {
      item.querySelector(".choice-name").append(node("span", "choice-confidence", percent(probabilities[name])));
    }
  });
  const answer = node("div", "jev-answer");
  answer.append(node("div", "choice-label", "JEV’s choice"));
  const row = node("div", "answer-row");
  row.append(
    node("span", "answer-name", pretty(payload.next_action)),
    node("span", "answer-confidence", percent(payload.confidence)),
  );
  answer.append(row);
  const args = payload.arguments || {};
  const entries = Object.entries(args).filter(([, value]) => value !== null && value !== "none" && value !== 0);
  if (entries.length) {
    const list = node("div", "argument-list");
    entries.forEach(([key, value]) => {
      const line = node("div", "argument");
      line.append(node("span", "", pretty(key)), node("span", "", String(value)));
      list.append(line);
    });
    answer.append(list);
  }
  card.append(answer);
  dom.decisionState.textContent = `Turn ${turn} · ${pretty(payload.next_action)}`;
}

function telemetry(kind, title, message, turn, argumentsValue) {
  dom.telemetryEmpty.classList.add("hidden");
  const card = node("section", `telemetry-card ${kind}`);
  const top = node("div", "telemetry-top");
  top.append(node("span", "", turn ? `TURN ${turn}` : "SESSION"), node("span", "", kind));
  card.append(top, node("h3", "telemetry-title", pretty(title)), node("p", "telemetry-message", message || ""));
  const entries = Object.entries(argumentsValue || {}).filter(([, value]) => value !== null && value !== "none" && value !== 0);
  if (entries.length) {
    const tags = node("div", "telemetry-args");
    entries.slice(0, 8).forEach(([key, value]) => tags.append(node("span", "tag", `${pretty(key)}: ${value}`)));
    card.append(tags);
  }
  dom.telemetryFeed.prepend(card);
}

function handleEvent(record) {
  const { event, payload = {} } = record;
  if (event === "session_started") {
    telemetry("running", "Fresh run started", payload.user_goal);
    dom.robotState.textContent = "Observing";
    setConnection("Robot session active", "busy");
  } else if (event === "given_that_ready") {
    renderContext(payload);
  } else if (event === "jev_question") {
    createTurn(payload);
  } else if (event === "jev_choice") {
    renderChoice(payload);
  } else if (event === "action_started") {
    telemetry("running", payload.action, "Driver is validating and executing this action.", payload.turn, payload.arguments);
    dom.robotState.textContent = `Executing · ${pretty(payload.action)}`;
  } else if (event === "action_finished") {
    telemetry(payload.success ? "success" : "failed", payload.action, payload.message, payload.turn, payload.data);
    dom.robotState.textContent = payload.success ? "Action verified" : "Action failed";
  } else if (event === "finish_rejected") {
    telemetry("failed", "Finish rejected", payload.message);
  } else if (event === "session_finished") {
    telemetry(payload.completed ? "success" : "failed", `Session ${payload.status}`, payload.message, null, { turns: payload.turns });
    dom.run.disabled = false;
    dom.robotState.textContent = pretty(payload.status);
    setConnection(payload.completed ? "Goal complete" : pretty(payload.status), payload.completed ? "ready" : "error");
  } else if (event === "session_error" || event === "session_failed") {
    telemetry("error", payload.type || payload.error_type || "Session error", payload.message);
    dom.run.disabled = false;
    dom.robotState.textContent = "Stopped";
    setConnection("Session stopped", "error");
  }
}

async function poll() {
  if (app.requestInFlight) return;
  app.requestInFlight = true;
  try {
    const response = await fetch(`/api/events?since=${app.cursor}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Event stream returned ${response.status}`);
    const update = await response.json();
    update.events.forEach(handleEvent);
    app.cursor = update.cursor;
    if (!update.running && dom.run.disabled && update.events.length === 0) dom.run.disabled = false;
  } catch (error) {
    setConnection("Interface disconnected", "error");
    dom.error.textContent = error.message;
  } finally {
    app.requestInFlight = false;
  }
}

dom.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const goal = dom.input.value.trim();
  if (!goal) return;
  dom.error.textContent = "";
  dom.run.disabled = true;
  resetWorkspace();
  setConnection("Starting fresh run", "busy");
  try {
    const response = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not start the robot session.");
    await poll();
  } catch (error) {
    dom.run.disabled = false;
    dom.error.textContent = error.message;
    setConnection("Could not start", "error");
  }
});

async function pollCamera() {
  try {
    const response = await fetch("/api/camera/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`Camera status returned ${response.status}`);
    const status = await response.json();
    dom.cameraMessage.textContent = status.message || "Waiting for wrist camera.";
    if (!status.available) {
      dom.cameraDot.className = "camera-dot";
      dom.cameraLabel.textContent = "Preview unavailable";
      dom.cameraSource.textContent = "Not exposed";
      return;
    }
    if (!status.connected || !status.frame_available) {
      dom.cameraDot.className = "camera-dot waiting";
      dom.cameraLabel.textContent = "Waiting for robot";
      dom.cameraSource.textContent = "Disconnected";
      dom.cameraFrame.classList.remove("live");
      dom.cameraOffline.classList.remove("hidden");
      dom.cameraLiveBadge.classList.remove("live");
      dom.cameraLiveBadge.textContent = "STANDBY";
      return;
    }
    dom.cameraDot.className = "camera-dot live";
    dom.cameraLabel.textContent = "Wrist RGB live";
    dom.cameraSource.textContent = pretty(status.source || "robot wrist rgb");
    dom.cameraLiveBadge.classList.add("live");
    dom.cameraLiveBadge.textContent = "LIVE";
    const shape = status.shape || [];
    dom.cameraResolution.textContent = shape.length === 2 ? `${shape[1]} × ${shape[0]}` : "LIVE";
    dom.cameraSequence.textContent = `FRAME ${String(status.sequence).padStart(6, "0")}`;
    if (status.sequence !== app.cameraSequence) {
      app.cameraSequence = status.sequence;
      dom.cameraFrame.src = `/api/camera/frame?sequence=${status.sequence}`;
    }
  } catch (error) {
    dom.cameraDot.className = "camera-dot";
    dom.cameraLabel.textContent = "Camera API offline";
    dom.cameraMessage.textContent = error.message;
  }
}

dom.cameraFrame.addEventListener("load", () => {
  dom.cameraFrame.classList.add("live");
  dom.cameraOffline.classList.add("hidden");
});

dom.cameraFrame.addEventListener("error", () => {
  dom.cameraFrame.classList.remove("live");
  dom.cameraOffline.classList.remove("hidden");
});

poll();
pollCamera();
setInterval(poll, 450);
setInterval(pollCamera, 700);
