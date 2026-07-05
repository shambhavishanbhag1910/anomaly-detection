const form =
    document.getElementById("anomalyForm");

const analyzeButton =
    document.getElementById("analyzeButton");

const buttonText =
    document.getElementById("buttonText");

const buttonLoader =
    document.getElementById("buttonLoader");

const sampleButton =
    document.getElementById("sampleButton");

const emptyState =
    document.getElementById("emptyState");

const resultsContainer =
    document.getElementById("resultsContainer");

const errorMessage =
    document.getElementById("errorMessage");


/* =========================
   SAMPLE MACHINE VALUES
========================= */

const normalSample = {
    airTemperature: 300.0,
    processTemperature: 310.0,
    rotationalSpeed: 1500,
    torque: 40.0,
    toolWear: 100
};


const anomalySample = {
    airTemperature: 303.4,
    processTemperature: 312.5,
    rotationalSpeed: 1200,
    torque: 68.0,
    toolWear: 230
};


let sampleToggle = false;


/* =========================
   LOAD SAMPLE
========================= */

sampleButton.addEventListener(
    "click",
    () => {

        sampleToggle = !sampleToggle;

        const sample =
            sampleToggle
                ? anomalySample
                : normalSample;


        document.getElementById(
            "airTemperature"
        ).value =
            sample.airTemperature;


        document.getElementById(
            "processTemperature"
        ).value =
            sample.processTemperature;


        document.getElementById(
            "rotationalSpeed"
        ).value =
            sample.rotationalSpeed;


        document.getElementById(
            "torque"
        ).value =
            sample.torque;


        document.getElementById(
            "toolWear"
        ).value =
            sample.toolWear;


        sampleButton.textContent =
            sampleToggle
                ? "Load Normal Sample"
                : "Load Anomaly Sample";
    }
);


/* =========================
   FORM SUBMIT
========================= */

form.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();

        clearError();

        setLoading(true);


        const payload = {

            "Air temperature [K]":
                Number(
                    document.getElementById(
                        "airTemperature"
                    ).value
                ),

            "Process temperature [K]":
                Number(
                    document.getElementById(
                        "processTemperature"
                    ).value
                ),

            "Rotational speed [rpm]":
                Number(
                    document.getElementById(
                        "rotationalSpeed"
                    ).value
                ),

            "Torque [Nm]":
                Number(
                    document.getElementById(
                        "torque"
                    ).value
                ),

            "Tool wear [min]":
                Number(
                    document.getElementById(
                        "toolWear"
                    ).value
                )
        };


        try {

            validatePayload(payload);


            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(payload)
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.error
                    || "Prediction request failed."
                );
            }


            renderResults(data);

        }

        catch (error) {

            showError(
                error.message
                || "Unable to analyze machine condition."
            );

        }

        finally {

            setLoading(false);

        }

    }
);


/* =========================
   VALIDATION
========================= */

function validatePayload(payload) {

    for (
        const [key, value]
        of Object.entries(payload)
    ) {

        if (
            typeof value !== "number"
            || Number.isNaN(value)
            || !Number.isFinite(value)
        ) {

            throw new Error(
                `Invalid value for ${key}`
            );
        }

    }

}


/* =========================
   RENDER RESULTS
========================= */

function renderResults(data) {

    emptyState.classList.add(
        "hidden"
    );

    resultsContainer.classList.remove(
        "hidden"
    );


    updatePriority(
        data.priority
    );


    updateModelCard(
        "iso",
        data.isolation_forest_anomaly,
        data.isolation_forest_score
    );


    updateModelCard(
        "ae",
        data.autoencoder_anomaly,
        data.reconstruction_error
    );


    updateAgreement(
        data.model_agreement,
        data.isolation_forest_anomaly,
        data.autoencoder_anomaly
    );


    updateRecommendation(
        data.priority
    );

}


/* =========================
   PRIORITY BANNER
========================= */

function updatePriority(priority) {

    const banner =
        document.getElementById(
            "priorityBanner"
        );

    const value =
        document.getElementById(
            "priorityValue"
        );

    const icon =
        document.getElementById(
            "priorityIcon"
        );


    banner.classList.remove(
        "normal",
        "medium",
        "high"
    );


    const normalizedPriority =
        String(priority).toLowerCase();


    banner.classList.add(
        normalizedPriority
    );


    value.textContent =
        priority;


    if (normalizedPriority === "high") {

        icon.textContent = "!";

    }

    else if (
        normalizedPriority === "medium"
    ) {

        icon.textContent = "△";

    }

    else {

        icon.textContent = "✓";

    }

}


/* =========================
   MODEL CARDS
========================= */

function updateModelCard(
    model,
    anomaly,
    score
) {

    const badge =
        document.getElementById(
            `${model}Badge`
        );

    const scoreElement =
        document.getElementById(
            `${model}Score`
        );

    const progress =
        document.getElementById(
            `${model}Progress`
        );


    const isAnomaly =
        Number(anomaly) === 1;


    badge.textContent =
        isAnomaly
            ? "Anomaly"
            : "Normal";


    badge.classList.remove(
        "normal",
        "anomaly"
    );


    badge.classList.add(
        isAnomaly
            ? "anomaly"
            : "normal"
    );


    scoreElement.textContent =
        formatScore(score);


    const progressWidth =
        getProgressWidth(
            Number(score),
            model
        );


    progress.style.width =
        `${progressWidth}%`;


    progress.style.background =
        isAnomaly
            ? "var(--high)"
            : "var(--normal)";

}


/* =========================
   SCORE FORMATTING
========================= */

function formatScore(score) {

    const numericScore =
        Number(score);


    if (!Number.isFinite(numericScore)) {

        return "-";

    }


    return numericScore.toFixed(4);

}


/* =========================
   PROGRESS VISUALIZATION
========================= */

function getProgressWidth(
    score,
    model
) {

    if (!Number.isFinite(score)) {

        return 0;

    }


    if (model === "iso") {

        /*
        Isolation Forest score may be
        negative or positive.

        This scaling is for UI visualization only.
        */

        const normalized =
            (score + 0.15) / 0.30;

        return clamp(
            normalized * 100,
            4,
            100
        );

    }


    /*
    Autoencoder reconstruction error.

    This is also visual scaling,
    not model threshold logic.
    */

    return clamp(
        score * 500,
        4,
        100
    );

}


function clamp(
    value,
    min,
    max
) {

    return Math.min(
        Math.max(value, min),
        max
    );

}


/* =========================
   MODEL AGREEMENT
========================= */

function updateAgreement(
    agreement,
    isoAnomaly,
    aeAnomaly
) {

    const value =
        document.getElementById(
            "agreementValue"
        );

    const description =
        document.getElementById(
            "agreementDescription"
        );


    if (agreement) {

        value.textContent =
            "Models Agree";


        if (
            Number(isoAnomaly) === 1
            &&
            Number(aeAnomaly) === 1
        ) {

            description.textContent =
                "Both models detected abnormal machine behavior. Investigation priority is strengthened.";

        }

        else {

            description.textContent =
                "Both models consider the current operating pattern consistent with normal behavior.";

        }

    }

    else {

        value.textContent =
            "Models Disagree";


        description.textContent =
            "Only one model detected abnormal behavior. Review the machine condition and continue monitoring.";

    }

}


/* =========================
   RECOMMENDATIONS
========================= */

function updateRecommendation(
    priority
) {

    const title =
        document.getElementById(
            "recommendationTitle"
        );

    const text =
        document.getElementById(
            "recommendationText"
        );


    const normalized =
        String(priority).toLowerCase();


    if (normalized === "high") {

        title.textContent =
            "Prioritize Maintenance Investigation";


        text.textContent =
            "Both anomaly detection models identified unusual machine behavior. Review current operating conditions, tool state, loading behavior, and recent maintenance history.";

    }

    else if (normalized === "medium") {

        title.textContent =
            "Review and Continue Monitoring";


        text.textContent =
            "One model detected unusual behavior. Review the current operating pattern and monitor the machine for continued or increasing deviation.";

    }

    else {

        title.textContent =
            "Continue Routine Monitoring";


        text.textContent =
            "Current operating conditions appear consistent with learned normal behavior. Continue standard monitoring and preventive maintenance practices.";

    }

}


/* =========================
   LOADING STATE
========================= */

function setLoading(isLoading) {

    analyzeButton.disabled =
        isLoading;


    if (isLoading) {

        buttonText.textContent =
            "Analyzing Machine...";


        buttonLoader.classList.remove(
            "hidden"
        );

    }

    else {

        buttonText.textContent =
            "Analyze Machine Condition";


        buttonLoader.classList.add(
            "hidden"
        );

    }

}


/* =========================
   ERROR HANDLING
========================= */

function showError(message) {

    errorMessage.textContent =
        message;


    errorMessage.classList.remove(
        "hidden"
    );

}


function clearError() {

    errorMessage.textContent = "";


    errorMessage.classList.add(
        "hidden"
    );

}