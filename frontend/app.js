"use strict";

const $ = (id) => document.getElementById(id);

let conversationId = null;
let previewUrl = null;

const translations = {
    en: {
        appTitle: "Smart Waste AI",
        appSubtitle: "Multimodal waste detection and recycling knowledge assistant",
        checkingApi: "Checking API...",
        apiConnected: "API connected",
        apiUnavailable: "API unavailable",
        conversation: "Conversation",
        newConversation: "New conversation",
        closeConversation: "Close",
        deleteConversation: "Delete",
        noConversation: "No active conversation",
        activeConversation: "Active conversation",
        analysis: "Analysis",
        workflow: "Workflow",
        workflowDetection: "Standalone Detection",
        workflowRetrieval: "Standalone Knowledge Retrieval",
        workflowCombined: "Detection-Based Retrieval",
        workflowImageQuestion: "Image + Question",
        descriptionDetection: "Upload an image to detect waste objects.",
        descriptionRetrieval: "Ask a knowledge question without uploading an image.",
        descriptionCombined: "Detect waste objects and retrieve knowledge using the detected classes.",
        descriptionImageQuestion: "Upload an image and ask a question about its contents. Vision and question-based retrieval run together.",
        uploadImage: "Upload image (JPEG / PNG, max 15 MiB)",
        yourQuestion: "Your question",
        questionPlaceholder: "Ask a question about the uploaded image or waste recycling...",
        responseLanguage: "Response language",
        generateAnswer: "Generate an answer using Gemini",
        generationHint: "Answer generation requires a configured Gemini API key. It is disabled by default.",
        runAnalysis: "Run analysis",
        processing: "Processing request...",
        analysisCompleted: "Analysis completed.",
        conversationCreated: "Conversation created.",
        conversationClosed: "Conversation closed.",
        conversationDeleted: "Conversation deleted.",
        selectImage: "Please select an image.",
        enterQuestion: "Please enter a question.",
        imageTooLarge: "Image exceeds the 15 MiB limit.",
        invalidImage: "Please select a JPEG or PNG image.",
        results: "Results",
        request: "Request",
        status: "Status",
        requestId: "Request ID",
        vision: "Vision Detection",
        imageDimensions: "Image dimensions",
        detections: "Detections",
        noDetections: "No objects detected.",
        retrieval: "Knowledge Retrieval",
        query: "Query",
        source: "Source",
        score: "Score",
        generatedAnswer: "Generated Answer",
        citation: "Citation",
        finalDecision: "Final Decision",
        action: "Action",
        reason: "Reason",
        reasonCode: "Reason code",
        clarification: "Clarification",
        warnings: "Warnings",
        errors: "Errors",
        removeImage: "Remove image",
        unavailable: "Unavailable",
        noAnswer: "No answer returned.",
        unknownError: "An unexpected error occurred.",
        languageChanged: "Language updated.",
        success: "Success",
        partialSuccess: "Partial success",
        failed: "Failed",
        needsClarification: "Needs clarification",
        noEvidence: "No evidence",
        insufficientEvidence: "Insufficient evidence",
        skipped: "Skipped",
        direct: "Direct answer",
        humanReview: "Human review",
        insufficient: "Insufficient evidence",
        cardboard: "Cardboard",
        glass: "Glass",
        metal: "Metal",
        paper: "Paper",
        plastic: "Plastic",
        generationUnavailable: "Answer generation is unavailable.",
        insufficientContext: "There is not enough information in the retrieved context to answer this question.",
        clarificationMessage: "I need a little more information to answer safely. Please clarify your request.",
        noUsableEvidence: "No usable retrieval evidence is available for generation."
    },

    fa: {
        appTitle: "هوش مصنوعی مدیریت پسماند",
        appSubtitle: "سامانه چندوجهی تشخیص پسماند و دستیار دانش بازیافت",
        checkingApi: "در حال بررسی ارتباط با API...",
        apiConnected: "ارتباط با API برقرار است",
        apiUnavailable: "API در دسترس نیست",
        conversation: "گفتگو",
        newConversation: "گفتگوی جدید",
        closeConversation: "بستن گفتگو",
        deleteConversation: "حذف گفتگو",
        noConversation: "گفتگوی فعالی وجود ندارد",
        activeConversation: "گفتگوی فعال",
        analysis: "تحلیل",
        workflow: "نوع عملیات",
        workflowDetection: "تشخیص مستقل پسماند",
        workflowRetrieval: "بازیابی مستقل دانش",
        workflowCombined: "بازیابی بر اساس تشخیص تصویر",
        workflowImageQuestion: "تصویر + سؤال",
        descriptionDetection: "برای شناسایی اشیای پسماند، یک تصویر بارگذاری کنید.",
        descriptionRetrieval: "بدون بارگذاری تصویر، یک سؤال دانشی درباره پسماند یا بازیافت بپرسید.",
        descriptionCombined: "اشیای پسماند شناسایی می‌شوند و بازیابی دانش بر اساس کلاس‌های تشخیص‌داده‌شده انجام می‌شود.",
        descriptionImageQuestion: "یک تصویر بارگذاری کنید و درباره محتوای آن سؤال بپرسید. تشخیص تصویر و بازیابی بر اساس سؤال شما هم‌زمان انجام می‌شوند.",
        uploadImage: "بارگذاری تصویر (JPEG / PNG، حداکثر ۱۵ مگابایت)",
        yourQuestion: "سؤال شما",
        questionPlaceholder: "درباره تصویر بارگذاری‌شده یا تفکیک و بازیافت پسماند سؤال بپرسید...",
        responseLanguage: "زبان رابط و پاسخ",
        generateAnswer: "تولید پاسخ با استفاده از Gemini",
        generationHint: "تولید پاسخ به کلید API جمینای نیاز دارد. این گزینه به‌صورت پیش‌فرض غیرفعال است.",
        runAnalysis: "شروع تحلیل",
        processing: "در حال پردازش درخواست...",
        analysisCompleted: "تحلیل با موفقیت انجام شد.",
        conversationCreated: "گفتگوی جدید ایجاد شد.",
        conversationClosed: "گفتگو بسته شد.",
        conversationDeleted: "گفتگو حذف شد.",
        selectImage: "لطفاً یک تصویر انتخاب کنید.",
        enterQuestion: "لطفاً سؤال خود را وارد کنید.",
        imageTooLarge: "حجم تصویر از محدودیت ۱۵ مگابایت بیشتر است.",
        invalidImage: "لطفاً یک تصویر JPEG یا PNG انتخاب کنید.",
        results: "نتایج",
        request: "درخواست",
        status: "وضعیت",
        requestId: "شناسه درخواست",
        vision: "نتیجه تشخیص تصویر",
        imageDimensions: "ابعاد تصویر",
        detections: "اشیای شناسایی‌شده",
        noDetections: "هیچ شیئی شناسایی نشد.",
        retrieval: "بازیابی دانش",
        query: "پرس‌وجو",
        source: "منبع",
        score: "امتیاز",
        generatedAnswer: "پاسخ تولیدشده",
        citation: "ارجاع",
        finalDecision: "تصمیم نهایی",
        action: "اقدام",
        reason: "دلیل",
        reasonCode: "کد دلیل",
        clarification: "نیاز به توضیح بیشتر",
        warnings: "هشدارها",
        errors: "خطاها",
        removeImage: "حذف تصویر",
        unavailable: "در دسترس نیست",
        noAnswer: "پاسخی دریافت نشد.",
        unknownError: "خطای پیش‌بینی‌نشده‌ای رخ داد.",
        languageChanged: "زبان رابط تغییر کرد.",
        success: "موفق",
        partialSuccess: "موفقیت نسبی",
        failed: "ناموفق",
        needsClarification: "نیازمند توضیح بیشتر",
        noEvidence: "بدون شواهد",
        insufficientEvidence: "شواهد ناکافی",
        skipped: "اجرا نشده",
        direct: "پاسخ مستقیم",
        humanReview: "بررسی انسانی",
        insufficient: "شواهد ناکافی",
        cardboard: "مقوا",
        glass: "شیشه",
        metal: "فلز",
        paper: "کاغذ",
        plastic: "پلاستیک",
        generationUnavailable: "تولید پاسخ در دسترس نیست.",
        insufficientContext: "اطلاعات موجود در منابع بازیابی‌شده برای پاسخ به این سؤال کافی نیست.",
        clarificationMessage: "برای پاسخ ایمن‌تر به اطلاعات بیشتری نیاز دارم. لطفاً درخواست خود را دقیق‌تر توضیح دهید.",
        noUsableEvidence: "شواهد بازیابی‌شده قابل استفاده‌ای برای تولید پاسخ وجود ندارد."
    }
};

function locale() {
    return $("locale").value;
}

function t(key) {
    const dictionary = translations[locale()];
    return dictionary[key] || translations.en[key] || key;
}

function localizeText(value) {
    if (typeof value !== "string") return value;

    const mappings = [
        ["I need a little more information to answer safely. Please clarify your request.", "clarificationMessage"],
        ["I don't have enough information in the retrieved context to answer this question.", "insufficientContext"],
        ["No usable retrieval evidence is available for generation.", "noUsableEvidence"],
        ["No answer returned.", "noAnswer"]
    ];

    for (const [original, key] of mappings) {
        if (value.includes(original)) {
            return value.replace(original, t(key));
        }
    }

    return value;
}

function applyLanguage() {
    const isPersian = locale() === "fa";

    document.documentElement.lang = isPersian ? "fa" : "en";
    document.documentElement.dir = isPersian ? "rtl" : "ltr";
    document.body.dir = isPersian ? "rtl" : "ltr";

    document.querySelectorAll("[data-i18n]").forEach(element => {
        const key = element.dataset.i18n;
        element.textContent = t(key);
    });

    document.querySelectorAll("[data-i18n-placeholder]").forEach(element => {
        element.placeholder = t(element.dataset.i18nPlaceholder);
    });

    updateConversation();
    updateWorkflow();

    if ($("health").dataset.connected === "true") {
        $("health").textContent = t("apiConnected");
    } else if ($("health").dataset.connected === "false") {
        $("health").textContent = t("apiUnavailable");
    }
}

function message(text, isError = false) {
    $("message").textContent = localizeText(text);
    $("message").className = isError ? "error-text" : "";
}

async function api(path, options = {}) {
    const response = await fetch(path, options);

    if (response.status === 204) return null;

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
        const detail = typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail || data);

        throw new Error(`HTTP ${response.status}: ${localizeText(detail)}`);
    }

    return data;
}

function updateConversation() {
    $("conversationInfo").textContent = conversationId
        ? `${t("activeConversation")}: ${conversationId}`
        : t("noConversation");

    $("closeBtn").disabled = !conversationId;
    $("deleteBtn").disabled = !conversationId;
}

function updateWorkflow() {
    const workflow = $("workflow").value;

    const needsImage = workflow !== "retrieval";
    const needsQuestion = workflow === "retrieval" ||
                          workflow === "image_question";

    $("imageGroup").hidden = !needsImage;
    $("questionGroup").hidden = !needsQuestion;

    const descriptions = {
        detection: "descriptionDetection",
        retrieval: "descriptionRetrieval",
        combined: "descriptionCombined",
        image_question: "descriptionImageQuestion"
    };

    $("workflowDescription").textContent =
        t(descriptions[workflow]);
}

async function checkHealth() {
    try {
        const result = await api("/health");
        const connected = result.status === "ok";

        $("health").dataset.connected = connected ? "true" : "false";
        $("health").textContent = connected
            ? t("apiConnected")
            : t("apiUnavailable");

        $("health").className = connected
            ? "status ok"
            : "status error";
    } catch {
        $("health").dataset.connected = "false";
        $("health").textContent = t("apiUnavailable");
        $("health").className = "status error";
    }
}

async function createConversation() {
    const result = await api("/api/v1/conversations", {
        method: "POST"
    });

    conversationId = result.conversation_id;
    updateConversation();
    message(t("conversationCreated"));
}

function addCard(title, lines) {
    const card = document.createElement("div");
    card.className = "result-card";

    const heading = document.createElement("h3");
    heading.textContent = title;
    card.appendChild(heading);

    for (const line of lines) {
        const paragraph = document.createElement("p");
        paragraph.textContent = line;
        card.appendChild(paragraph);
    }

    $("resultContent").appendChild(card);
}

function translateStatus(status) {
    const values = {
        success: "success",
        partial_success: "partialSuccess",
        failed: "failed",
        needs_clarification: "needsClarification",
        no_detections: "noDetections",
        no_evidence: "noEvidence",
        insufficient_evidence: "insufficientEvidence",
        skipped: "skipped",
        DIRECT: "direct",
        HUMAN_REVIEW: "humanReview",
        INSUFFICIENT_EVIDENCE: "insufficient",
        DECIDED: "success",
        UNAVAILABLE: "unavailable"
    };

    return values[status] ? t(values[status]) : (status ?? t("unavailable"));
}

function translateClassName(name) {
    const values = {
        cardboard: "cardboard",
        glass: "glass",
        metal: "metal",
        paper: "paper",
        plastic: "plastic"
    };

    return values[name] ? t(values[name]) : name;
}

function renderResults(data) {
    const root = $("resultContent");
    root.replaceChildren();
    $("results").hidden = false;

    addCard(t("request"), [
        `${t("status")}: ${translateStatus(data.status)}`,
        `${t("requestId")}: ${data.request_id}`
    ]);

    if (data.vision) {
        const detections = Array.isArray(data.vision.detections)
            ? data.vision.detections
            : [];

        const lines = [
            `${t("status")}: ${translateStatus(data.vision.status)}`,
            `${t("imageDimensions")}: ${data.vision.image_width} × ${data.vision.image_height}`
        ];

        if (detections.length === 0) {
            lines.push(t("noDetections"));
        } else {
            detections.forEach(detection => {
                lines.push(
                    `${translateClassName(detection.class_name)} — ${(detection.confidence * 100).toFixed(1)}% | Box: ${detection.bbox_xyxy.join(", ")}`
                );
            });
        }

        addCard(t("vision"), lines);
    }

    if (data.retrieval) {
        const lines = [
            `${t("status")}: ${translateStatus(data.retrieval.status)}`,
            `${t("query")}: ${data.retrieval.query}`
        ];

        const results = Array.isArray(data.retrieval.results)
            ? data.retrieval.results
            : [];

        results.forEach(result => {
            lines.push(
                `#${result.rank} | ${t("source")}: ${result.source} | ${t("score")}: ${Number(result.score).toFixed(4)}\n${result.text}`
            );
        });

        addCard(t("retrieval"), lines);
    }

    if (data.generation) {
        const lines = [
            `${t("status")}: ${translateStatus(data.generation.status)}`,
            localizeText(
                data.generation.answer ||
                data.generation.reason ||
                t("noAnswer")
            )
        ];

        (data.generation.citation_chunk_ids || []).forEach(id => {
            lines.push(`${t("citation")}: ${id}`);
        });

        addCard(t("generatedAnswer"), lines);
    }

    if (data.final_decision) {
        addCard(t("finalDecision"), [
            `${t("action")}: ${translateStatus(data.final_decision.action)}`,
            `${t("reason")}: ${localizeText(data.final_decision.reason || "")}`,
            `${t("reasonCode")}: ${data.final_decision.reason_code || ""}`
        ]);
    }

    if (data.clarification_message) {
        addCard(t("clarification"), [
            localizeText(data.clarification_message)
        ]);
    }

    if (data.warnings?.length) {
        addCard(
            t("warnings"),
            data.warnings.map(value => `⚠ ${localizeText(value)}`)
        );
    }

    if (data.errors?.length) {
        addCard(
            t("errors"),
            data.errors.map(error =>
                `${error.stage}: ${localizeText(error.message)}`
            )
        );
    }

    $("results").scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

function clearImagePreview() {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    $("imagePreview").removeAttribute("src");
    $("imagePreviewContainer").hidden = true;
}

function updateImagePreview() {
    const file = $("image").files[0];

    clearImagePreview();

    if (!file) return;

    previewUrl = URL.createObjectURL(file);
    $("imagePreview").src = previewUrl;
    $("imagePreviewContainer").hidden = false;
}

async function analyze() {
    $("analyzeBtn").disabled = true;

    try {
        if (!conversationId) {
            await createConversation();
        }

        const workflow = $("workflow").value;
        const image = $("image").files[0];
        const question = $("question").value.trim();

        const needsImage = workflow !== "retrieval";
        const needsQuestion = workflow === "retrieval" ||
                              workflow === "image_question";

        if (needsImage && !image) {
            throw new Error(t("selectImage"));
        }

        if (needsQuestion && !question) {
            throw new Error(t("enterQuestion"));
        }

        if (image && image.size > 15 * 1024 * 1024) {
            throw new Error(t("imageTooLarge"));
        }

        if (image && !["image/jpeg", "image/png"].includes(image.type)) {
            throw new Error(t("invalidImage"));
        }

        const options = {
            run_vision: needsImage,
            run_retrieval: workflow !== "detection",
            generate_answer: $("generate").checked,
            retrieval_query_source: workflow === "combined"
                ? "vision_detections"
                : "question"
        };

        const form = new FormData();

        form.append("request_id", crypto.randomUUID());

        form.append(
            "task_type",
            workflow === "retrieval"
                ? "knowledge_question"
                : "detection"
        );

        form.append("locale", locale());
        form.append("options", JSON.stringify(options));

        if (needsQuestion) {
            form.append("question", question);
        }

        if (needsImage) {
            form.append("image", image);
        }

        message(t("processing"));

        const result = await api(
            `/api/v1/conversations/${encodeURIComponent(conversationId)}/analyze`,
            {
                method: "POST",
                body: form
            }
        );

        renderResults(result);
        message(t("analysisCompleted"));

    } catch (error) {
        message(error.message || t("unknownError"), true);
    } finally {
        $("analyzeBtn").disabled = false;
    }
}

$("workflow").addEventListener("change", updateWorkflow);

$("locale").addEventListener("change", () => {
    applyLanguage();
});

$("image").addEventListener("change", updateImagePreview);

$("removeImageBtn").addEventListener("click", () => {
    $("image").value = "";
    clearImagePreview();
});

$("createBtn").addEventListener("click", async () => {
    $("createBtn").disabled = true;

    try {
        await createConversation();
    } catch (error) {
        message(error.message, true);
    } finally {
        $("createBtn").disabled = false;
    }
});

$("analyzeBtn").addEventListener("click", analyze);

$("closeBtn").addEventListener("click", async () => {
    $("closeBtn").disabled = true;

    try {
        await api(
            `/api/v1/conversations/${encodeURIComponent(conversationId)}/close`,
            { method: "POST" }
        );

        conversationId = null;
        updateConversation();
        message(t("conversationClosed"));

    } catch (error) {
        message(error.message, true);
    }
});

$("deleteBtn").addEventListener("click", async () => {
    $("deleteBtn").disabled = true;

    try {
        await api(
            `/api/v1/conversations/${encodeURIComponent(conversationId)}`,
            { method: "DELETE" }
        );

        conversationId = null;
        updateConversation();
        message(t("conversationDeleted"));

    } catch (error) {
        message(error.message, true);
    }
});

applyLanguage();
updateConversation();
checkHealth();
