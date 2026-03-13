/* ──────────────────────────────────────────────────────────────
   DSA Tutor — Frontend Logic (vanilla JS, no framework)
   ────────────────────────────────────────────────────────────── */

(function () {
    "use strict";

    // ── DOM refs ───────────────────────────────────────────────
    const chatContainer  = document.getElementById("chat-container");
    const questionInput  = document.getElementById("question-input");
    const sendBtn        = document.getElementById("send-btn");
    const loadingEl      = document.getElementById("loading-indicator");
    const ratingEl       = document.getElementById("skill-rating");
    const matchesEl      = document.getElementById("skill-matches");
    const usernameInput  = document.getElementById("username");
    const topicSelect    = document.getElementById("topic");

    // ── State ──────────────────────────────────────────────────
    let isBusy = false;

    // ── Helpers ────────────────────────────────────────────────

    function getUsername() {
        return (usernameInput.value || "student").trim();
    }

    function getTopic() {
        return topicSelect.value;
    }

    function setLoading(on) {
        isBusy = on;
        sendBtn.disabled = on;
        loadingEl.classList.toggle("hidden", !on);
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function appendMessage(html, className) {
        const div = document.createElement("div");
        div.className = "message " + className;
        div.innerHTML = html;
        chatContainer.appendChild(div);
        scrollToBottom();
        return div;
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function updateSkillDisplay(rating, matches) {
        ratingEl.textContent = Math.round(rating);
        matchesEl.textContent = "(" + matches + " matches)";
    }

    // ── API calls ──────────────────────────────────────────────

    async function apiPost(path, body) {
        const res = await fetch(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const err = await res.json().catch(function () {
                return { detail: "Request failed (" + res.status + ")" };
            });
            throw new Error(err.detail || "Request failed");
        }
        return res.json();
    }

    async function apiGet(path) {
        const res = await fetch(path);
        if (!res.ok) {
            throw new Error("Request failed (" + res.status + ")");
        }
        return res.json();
    }

    async function fetchSkills() {
        try {
            var data = await apiGet("/skills/" + encodeURIComponent(getUsername()));
            var topic = getTopic();
            var found = false;
            for (var i = 0; i < data.skills.length; i++) {
                if (data.skills[i].topic === topic) {
                    updateSkillDisplay(data.skills[i].rating, data.skills[i].matches);
                    found = true;
                    break;
                }
            }
            if (!found) {
                updateSkillDisplay(1000, 0);
            }
        } catch (e) {
            // Silently ignore — skill display will show defaults
        }
    }

    // ── Send question ──────────────────────────────────────────

    async function sendQuestion() {
        var question = questionInput.value.trim();
        if (!question || isBusy) return;

        // Show user message
        appendMessage(escapeHtml(question), "user-message");
        questionInput.value = "";
        setLoading(true);

        try {
            var data = await apiPost("/ask", {
                username: getUsername(),
                topic: getTopic(),
                question: question,
                difficulty: 1000.0,
            });

            // Update skill display
            updateSkillDisplay(data.rating, data.matches);

            // Build assistant message
            var content = "";
            if (data.rag_used) {
                content += '<span class="rag-badge">RAG context used</span><br>';
            }
            content += escapeHtml(data.response);

            var msgDiv = appendMessage(content, "assistant-message");

            // Add feedback buttons
            var feedbackRow = document.createElement("div");
            feedbackRow.className = "feedback-row";

            var correctBtn = document.createElement("button");
            correctBtn.className = "feedback-btn";
            correctBtn.textContent = "\u2714 Helpful";
            correctBtn.setAttribute("data-iid", data.interaction_id);

            var incorrectBtn = document.createElement("button");
            incorrectBtn.className = "feedback-btn";
            incorrectBtn.textContent = "\u2718 Not helpful";
            incorrectBtn.setAttribute("data-iid", data.interaction_id);

            feedbackRow.appendChild(correctBtn);
            feedbackRow.appendChild(incorrectBtn);
            msgDiv.appendChild(feedbackRow);

            // Feedback handlers
            correctBtn.addEventListener("click", function () {
                submitFeedback(data.interaction_id, true, correctBtn, incorrectBtn, msgDiv);
            });
            incorrectBtn.addEventListener("click", function () {
                submitFeedback(data.interaction_id, false, incorrectBtn, correctBtn, msgDiv);
            });

        } catch (err) {
            appendMessage(escapeHtml("Error: " + err.message), "error-message");
        } finally {
            setLoading(false);
        }
    }

    // ── Submit feedback ────────────────────────────────────────

    async function submitFeedback(interactionId, correct, activeBtn, otherBtn, msgDiv) {
        activeBtn.disabled = true;
        otherBtn.disabled = true;

        if (correct) {
            activeBtn.classList.add("selected-correct");
        } else {
            activeBtn.classList.add("selected-incorrect");
        }

        try {
            var data = await apiPost("/feedback", {
                interaction_id: interactionId,
                username: getUsername(),
                topic: getTopic(),
                difficulty: 1000.0,
                answered_correctly: correct,
            });

            updateSkillDisplay(data.rating_after, data.matches);

            // Show delta
            var deltaDiv = document.createElement("div");
            deltaDiv.className = "delta-display";
            var sign = data.delta >= 0 ? "+" : "";
            deltaDiv.textContent = sign + data.delta.toFixed(1) + " ELO";
            deltaDiv.classList.add(data.delta >= 0 ? "delta-positive" : "delta-negative");
            msgDiv.appendChild(deltaDiv);

        } catch (err) {
            appendMessage(escapeHtml("Feedback error: " + err.message), "error-message");
            activeBtn.disabled = false;
            otherBtn.disabled = false;
            activeBtn.classList.remove("selected-correct", "selected-incorrect");
        }
    }

    // ── Event listeners ────────────────────────────────────────

    sendBtn.addEventListener("click", sendQuestion);

    questionInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });

    // Refresh skill display when user or topic changes
    usernameInput.addEventListener("change", fetchSkills);
    topicSelect.addEventListener("change", fetchSkills);

    // Initial skill fetch
    fetchSkills();

})();
