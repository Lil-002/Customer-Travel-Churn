console.log('Customer Churn Predictor JavaScript loaded');

// Global state
let currentPrediction = null;

// DOM Elements
const predictButton = document.getElementById('predictButton');
const loadSampleBtn = document.getElementById('loadSampleBtn');
const resultsCard = document.getElementById('resultsCard');
const welcomeCard = document.getElementById('welcomeCard');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded and parsed');
    setupEventListeners();
    checkModelStatus();
});

function setupEventListeners() {
    console.log('Setting up event listeners...');

    // Predict button
    if (predictButton) {
        console.log('Found predict button');
        predictButton.addEventListener('click', handleFormSubmit);
    } else {
        console.error('Predict button not found! Looking for #predictButton');
    }

    // Load sample data button
    if (loadSampleBtn) {
        loadSampleBtn.addEventListener('click', loadSampleData);
    }

    // Show model info button
    const modelInfoBtn = document.getElementById('modelInfoBtn');
    if (modelInfoBtn) {
        modelInfoBtn.addEventListener('click', showModelInfo);
    }

     // Results card buttons
    const newPredictionBtn = document.getElementById('newPredictionBtn');
    if (newPredictionBtn) {
        newPredictionBtn.addEventListener('click', resetForm);
    }

     // Prevent form submission on Enter key
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(event) {
            event.preventDefault();
            console.log('Form submit prevented');
            return false;
        });

        // Also prevent any form submissions
        predictionForm.onsubmit = function() {
            console.log('Form onsubmit prevented');
            return false;
        };
    }
}

async function checkModelStatus() {
    console.log('Checking model status...');
    try {
        const response = await fetch('/model/info');
        console.log('Model info response status:', response.status);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Model info data:', data);

        if (data.success) {
            updateModelStatus(true, data.model_info.best_model, data.shap_available);
        } else {
            updateModelStatus(false, 'Not loaded - ' + (data.error || 'Unknown error'), false);
        }
    } catch (error) {
        console.error('Failed to check model status:', error);
        updateModelStatus(false, 'Connection error', false);
    }
}

function updateModelStatus(isLoaded, modelName, shapAvailable) {
    const statusElement = document.getElementById('modelStatus');
    const statusText = document.getElementById('modelStatusText');

    console.log(`Updating model status: loaded=${isLoaded}, name=${modelName}`);

    if (statusElement && statusText) {
        if (isLoaded) {
            statusElement.className = 'model-status model-active';
            statusElement.innerHTML = `<i class="fas fa-check-circle"></i> ${modelName}`;
            statusText.textContent = 'Model Active';

            // Update SHAP badge
            if (shapAvailable) {
                const shapBadge = document.getElementById('shapBadge');
                if (shapBadge) {
                    shapBadge.innerHTML = '<i class="fas fa-brain"></i> SHAP Enabled';
                    shapBadge.className = 'badge bg-info ms-2';
                }
            }
        } else {
            statusElement.className = 'model-status model-inactive';
            statusElement.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${modelName}`;
            statusText.textContent = modelName || 'Model Not Ready';
        }
    }
}

async function handleFormSubmit(event) {
    console.log('handleFormSubmit called');

    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }

    // Collect form data with CORRECT CSV column names
    const formData = {
        Ages: document.getElementById('age').value,
        FrequentFlyer: document.getElementById('frequent_flyer').value,
        AnnualIncomeClass: document.getElementById('income_class').value,
        ServicesOpted: document.getElementById('services_opted').value,
        AccountSyncedToSocialMedia: document.getElementById('social_sync').value,
        BookedHotelOrNot: document.getElementById('booked_hotel').value
    };

    console.log('Form data:', formData);

    // Validate form data
    if (!validateFormData(formData)) {
        showNotification('Please fill in all fields correctly', 'warning');
        return false;
    }

    // Convert numeric fields
    formData.Ages = parseInt(formData.Ages);
    formData.ServicesOpted = parseInt(formData.ServicesOpted);

    // Show loading
    showLoading('Predicting churn probability...');

    try {
        console.log('Sending request to /predict...');

        // First check if server is reachable
        const healthCheck = await fetch('/health');
        if (!healthCheck.ok) {
            throw new Error('Server is not responding');
        }

        // Send prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        console.log('Response received, status:', response.status);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const result = await response.json();
        console.log('Prediction result:', result);

        if (result.success) {
            // Store prediction
            currentPrediction = result;

            // Update UI with results
            updateResultsUI(result);

            // Show results card
            showResultsCard();

            showNotification('Prediction completed successfully!', 'success');
        } else {
            showNotification(`Prediction failed: ${result.error}`, 'danger');
            console.error('Prediction error:', result.error);
        }

    } catch (error) {
        console.error('Prediction error:', error);
        showNotification(`Failed: ${error.message}`, 'danger');
    } finally {
        hideLoading();
    }

    return false;
}

function validateFormData(data) {
    console.log('Validating form data:', data);

    // Check all fields are filled
    for (const [key, value] of Object.entries(data)) {
        if (!value || value.toString().trim() === '') {
            console.log(`Validation failed: ${key} is empty`);
            return false;
        }
    }

    // Validate age (Ages)
    const age = parseInt(data.Ages);
    if (isNaN(age) || age < 18 || age > 100) {
        console.log('Validation failed: Ages must be between 18 and 100');
        return false;
    }

    // Validate services (ServicesOpted)
    const services = parseInt(data.ServicesOpted);
    if (isNaN(services) || services < 1 || services > 6) {
        console.log('Validation failed: ServicesOpted must be between 1 and 6');
        return false;
    }

    // Validate Frequent Flyer (must be one of three options)
    const validFrequentFlyer = ['Yes', 'No', 'No Record'];
    if (!validFrequentFlyer.includes(data.FrequentFlyer)) {
        console.log('Validation failed: FrequentFlyer must be Yes, No, or No Record');
        return false;
    }

    return true;
}

function showLoading(message) {
    if (loadingOverlay && loadingText) {
        loadingText.textContent = message || 'Processing...';
        loadingOverlay.style.display = 'flex';
    }
}

function hideLoading() {
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
}

function showNotification(message, type) {
    // Remove any existing notifications
    const existingNotifications = document.querySelectorAll('.notification-toast');
    existingNotifications.forEach(notification => notification.remove());

    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification-toast alert alert-${type} alert-dismissible fade show`;
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;

    // Style the notification
    notification.style.position = 'fixed';
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '9999';
    notification.style.minWidth = '300px';
    notification.style.maxWidth = '500px';

    // Add to page
    document.body.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

function updateResultsUI(result) {
    // Update probability circle
    const probability = result.prediction.probability * 100;
    const probabilityCircle = document.getElementById('probabilityCircle');
    const probabilityValue = document.getElementById('probabilityValue');

    if (probabilityCircle && probabilityValue) {
        probabilityCircle.style.setProperty('--percentage', probability);
        probabilityCircle.style.setProperty('--circle-color', result.prediction.risk_color);
        probabilityValue.textContent = `${probability.toFixed(1)}%`;
    }

    // Update risk badge
    const riskBadge = document.getElementById('riskBadge');
    if (riskBadge) {
        riskBadge.textContent = result.prediction.risk_level;
        riskBadge.className = 'risk-badge mb-2';
        riskBadge.style.backgroundColor = result.prediction.risk_color;
        riskBadge.style.color = 'white';
    }

    // Update recommendation
    const recommendationAlert = document.getElementById('recommendationAlert');
    const recommendationText = document.getElementById('recommendationText');
    if (recommendationAlert && recommendationText) {
        recommendationAlert.className = `alert alert-${getAlertType(result.prediction.risk_level)}`;
        recommendationText.textContent = result.prediction.recommendation;
    }

    // Update model info
    document.getElementById('modelName').textContent = result.model_info.name;
    document.getElementById('confidenceLevel').textContent = `${(result.model_info.accuracy * 100).toFixed(1)}%`;
    document.getElementById('predictionTime').textContent = new Date(result.timestamp).toLocaleTimeString();

    // Update customer summary
    document.getElementById('summaryAge').textContent = result.customer.age;
    document.getElementById('summaryFlyer').textContent = result.customer.frequent_flyer;
    document.getElementById('summaryIncome').textContent = result.customer.income_class;
    document.getElementById('summaryServices').textContent = result.customer.services_opted;
    document.getElementById('summarySocial').textContent = result.customer.social_sync;
    document.getElementById('summaryHotel').textContent = result.customer.booked_hotel;

    // Update actionable insights
    updateRecommendationsList(result.actionable_insights);
    // Update recommendations list
    updateRecommendationsList(result.recommendations);

    // Update risk factors summary:
    // Normalize contributions from various possible response shapes then call updateRiskFactorsSummary
    try {
        // Normalize and choose the contributions array from server response
        let contributions = [];

        if (result && result.shap && Array.isArray(result.shap.contributions)) {
            contributions = result.shap.contributions;
        }
        else if (result && Array.isArray(result.shap_contributions)) {
            contributions = result.shap_contributions;
        }
        // Structured risk_factors (combine increasing + decreasing into one contributions-like array)
        else if (result && result.risk_factors && (Array.isArray(result.risk_factors.increasing) || Array.isArray(result.risk_factors.decreasing))) {
            const inc = result.risk_factors.increasing || [];
            const dec = result.risk_factors.decreasing || [];
            // Keep sign and abs_contribution consistent
            contributions = inc.concat(dec);
        }
        // Legacy: result.contributions
        else if (result && Array.isArray(result.contributions)) {
            contributions = result.contributions;
        }

        // Ensure each contribution has required fields (simplified_name, contribution, abs_contribution)
        contributions = contributions.map(c => {
            const contributionVal = (typeof c.contribution === 'number') ? c.contribution : (typeof c.value === 'number' ? c.value : 0);
            const absVal = (typeof c.abs_contribution === 'number') ? c.abs_contribution : Math.abs(contributionVal);
            return {
                feature: c.feature || c.name || 'unknown',
                simplified_name: c.simplified_name || c.display_name || c.feature || 'unknown',
                contribution: contributionVal,
                abs_contribution: absVal,
                reason: c.reason || c.explanation || ''
            };
        });
    // Update risk factors summary:
    // Pass the entire contribution so the summary function can handle either 'contributions' or
    // the server-side 'risk_factors', 'key_takeaways', 'actionable_insights'.
    updateRiskFactorsSummary(contributions);
} catch (err) {
    console.error("Failed to update risk factors summary:", err);
}}

function getAlertType(riskLevel) {
    switch(riskLevel) {
        case 'VERY HIGH': return 'danger';
        case 'HIGH': return 'warning';
        case 'MEDIUM': return 'info';
        case 'LOW': return 'success';
        default: return 'secondary';
    }
}

function showResultsCard() {
    if (welcomeCard) welcomeCard.style.display = 'none';
    if (resultsCard) resultsCard.style.display = 'block';
}

function loadSampleData() {
    console.log('Loading sample data...');

    // Sample customer data with CORRECT CSV column names
    const sampleData = {
        Ages: 45,
        FrequentFlyer: 'Yes',
        AnnualIncomeClass: 'High Income',
        ServicesOpted: 5,
        AccountSyncedToSocialMedia: 'Yes',
        BookedHotelOrNot: 'No'
    };

    // Fill form with sample data
    document.getElementById('age').value = sampleData.Ages;
    document.getElementById('frequent_flyer').value = sampleData.FrequentFlyer;
    document.getElementById('income_class').value = sampleData.AnnualIncomeClass;
    document.getElementById('services_opted').value = sampleData.ServicesOpted;
    document.getElementById('social_sync').value = sampleData.AccountSyncedToSocialMedia;
    document.getElementById('booked_hotel').value = sampleData.BookedHotelOrNot;

    showNotification('Sample customer data loaded!', 'info');
}

function showModelInfo() {
    fetch('/model/info')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const info = data.model_info;
                const metrics = info.best_metrics || {};

                const message = `
                    Model: ${info.best_model}

                    Performance Metrics:
                    • Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%
                    • F1 Score: ${(metrics.f1 * 100).toFixed(1)}%
                    • Precision: ${(metrics.precision * 100).toFixed(1)}%
                    • Recall: ${(metrics.recall * 100).toFixed(1)}%

                    Status: ${data.loaded ? 'Loaded and ready' : 'Not loaded'}
                    SHAP Available: ${data.shap_available ? 'Yes' : 'No'}
                `;

                alert(message);
            } else {
                showNotification('Could not load model info: ' + data.error, 'warning');
            }
        })
        .catch(error => {
            console.error('Error fetching model info:', error);
            showNotification('Failed to load model info', 'danger');
        });
}

function updateRecommendationsList(recommendations) {
    const insightsContainer = document.getElementById('insightsContainer');
    if (!insightsContainer) return;
        // Safety check
    if (!Array.isArray(recommendations)) {
        insightsContainer.innerHTML = `
            <div class="text-muted small">No recommendations available.</div>
        `;
        return;
    }

    let html = '<div class="row">';

    recommendations.forEach((rec, index) => {
        const colors = ['primary', 'success', 'info', 'warning', 'danger'];
        const color = colors[index % colors.length];
        //const text = (typeof rec === 'string') ? rec : (rec.text || JSON.stringify(rec));

        html += `
            <div class="col-md-6 mb-3">
                <div class="insight-card">
                    <div class="insight-icon bg-${color}">
                        <i class="fas fa-lightbulb"></i>
                    </div>
                    <div class="insight-content">
                        <h6>Recommendation ${index + 1}</h6>
                        <p class="small mb-0">${rec}</p>
                    </div>
                </div>
            </div>
        `;
    });

    html += '</div>';
    insightsContainer.innerHTML = html;
}

// Handling Risk Factors summary update
function updateRiskFactorsSummary(contributions) {
    console.log('Updating risk factors summary...');
    
    const increasingFactorsContainer = document.getElementById('increasingFactorsList');
    const decreasingFactorsContainer = document.getElementById('decreasingFactorsList');
    const takeawaysList = document.getElementById('takeawaysList');
    const riskFactorsSection = document.getElementById('riskFactorsSummary');
    
    if (!riskFactorsSection) {
        console.error('Risk factors section not found');
        return;
    }
    
    // Show the section
    riskFactorsSection.style.display = 'block';
    
    // Separate increasing and decreasing factors
    const increasingFactors = contributions.filter(c => c.contribution > 0);
    const decreasingFactors = contributions.filter(c => c.contribution < 0);
    
    // Update increasing factors (risk factors)
    if (increasingFactorsContainer) {
        if (increasingFactors.length > 0) {
            let html = '<div class="list-group list-group-flush">';
            increasingFactors.slice(0, 5).forEach((factor, index) => {
                const impactStrength = getImpactStrength(factor.abs_contribution);
                html += `
                    <div class="list-group-item border-0 py-2">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <span class="badge bg-danger me-2">${index + 1}</span>
                                <span class="fw-semibold">${factor.simplified_name}</span>
                            </div>
                            <div>
                                <span class="text-danger fw-bold">+${factor.contribution.toFixed(3)}</span>
                                <span class="badge ${getImpactBadgeClass(impactStrength)} ms-2">
                                    ${impactStrength}
                                </span>
                            </div>
                        </div>
                        <div class="small text-muted mt-1">
                            <i class="fas fa-info-circle me-1"></i>
                            ${getRiskFactorExplanation(factor.simplified_name, factor.contribution)}
                        </div>
                    </div>
                `;
            });
            html += '</div>';
            increasingFactorsContainer.innerHTML = html;
        } else {
            increasingFactorsContainer.innerHTML = `
                <div class="text-center py-3">
                    <i class="fas fa-check-circle fa-2x text-success mb-2"></i>
                    <p class="text-muted">No significant risk-increasing factors identified</p>
                </div>
            `;
        }
    }
    
    // Update decreasing factors (protective factors)
    if (decreasingFactorsContainer) {
        if (decreasingFactors.length > 0) {
            let html = '<div class="list-group list-group-flush">';
            decreasingFactors.slice(0, 5).forEach((factor, index) => {
                const impactStrength = getImpactStrength(factor.abs_contribution);
                html += `
                    <div class="list-group-item border-0 py-2">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <span class="badge bg-success me-2">${index + 1}</span>
                                <span class="fw-semibold">${factor.simplified_name}</span>
                            </div>
                            <div>
                                <span class="text-success fw-bold">${factor.contribution.toFixed(3)}</span>
                                <span class="badge ${getImpactBadgeClass(impactStrength)} ms-2">
                                    ${impactStrength}
                                </span>
                            </div>
                        </div>
                        <div class="small text-muted mt-1">
                            <i class="fas fa-info-circle me-1"></i>
                            ${getProtectiveFactorExplanation(factor.simplified_name, factor.contribution)}
                        </div>
                    </div>
                `;
            });
            html += '</div>';
            decreasingFactorsContainer.innerHTML = html;
        } else {
            decreasingFactorsContainer.innerHTML = `
                <div class="text-center py-3">
                    <i class="fas fa-info-circle fa-2x text-warning mb-2"></i>
                    <p class="text-muted">No significant protective factors identified</p>
                </div>
            `;
        }
    }
    
    // Update key takeaways
    if (takeawaysList) {
        let takeaways = generateKeyTakeaways(contributions, increasingFactors, decreasingFactors);
        takeawaysList.innerHTML = '';
        
        takeaways.forEach(takeaway => {
            const li = document.createElement('li');
            li.className = 'mb-2';
            li.innerHTML = `
                <i class="fas ${takeaway.icon} me-2 text-${takeaway.color}"></i>
                ${takeaway.text}
            `;
            takeawaysList.appendChild(li);
        });
    }
}

function getImpactStrength(absContribution) {
    if (absContribution > 0.1) return 'HIGH';
    if (absContribution > 0.05) return 'MEDIUM';
    if (absContribution > 0.01) return 'LOW';
    return 'MINIMAL';
}

function getImpactBadgeClass(impactStrength) {
    switch(impactStrength) {
        case 'HIGH': return 'bg-danger';
        case 'MEDIUM': return 'bg-warning text-dark';
        case 'LOW': return 'bg-info';
        default: return 'bg-secondary';
    }
}

function getRiskFactorExplanation(featureName, contribution) {
    const explanations = {
        'Age': `Older age groups show ${contribution > 0.05 ? 'significantly ' : ''}higher churn rates`,
        'Frequent Flyer': 'Frequent flyers are more likely to explore competitor offers',
        'Income': 'Income level strongly influences service affordability and retention',
        'Services': 'Number of services impacts overall customer commitment',
        'Social': 'Social media connected customers are more exposed to competitor marketing',
        'Hotel': 'Hotel booking history indicates engagement with travel services'
    };
    
    // Check for partial matches
    featureName = featureName.toLowerCase();
    for (const [key, explanation] of Object.entries(explanations)) {
        if (featureName.includes(key.toLowerCase())) {
            return explanation;
        }
    }
    
    return 'This feature significantly impacts churn prediction';
}

function getProtectiveFactorExplanation(featureName, contribution) {
    const explanations = {
        'Age': 'Age group shows lower churn propensity',
        'Frequent Flyer': 'Loyalty program participation increases retention',
        'Income': 'Income stability supports service continuity',
        'Services': 'Service bundle reduces likelihood of switching',
        'Social': 'Limited social exposure reduces competitive pressure',
        'Hotel': 'Travel service engagement indicates satisfaction'
    };
    
    // Check for partial matches
    featureName = featureName.toLowerCase();
    for (const [key, explanation] of Object.entries(explanations)) {
        if (featureName.includes(key.toLowerCase())) {
            return explanation;
        }
    }
    
    return 'This feature reduces churn risk probability';
}

function generateKeyTakeaways(allContributions, increasingFactors, decreasingFactors) {
    const takeaways = [];
    
    // Overall risk assessment
    const totalRisk = increasingFactors.reduce((sum, f) => sum + f.abs_contribution, 0);
    const totalProtection = decreasingFactors.reduce((sum, f) => sum + Math.abs(f.contribution), 0);
    
    if (totalRisk > totalProtection * 1.5) {
        takeaways.push({
            icon: 'fa-exclamation-triangle',
            color: 'danger',
            text: 'Customer has significantly more risk factors than protective factors'
        });
    } else if (totalProtection > totalRisk * 1.5) {
        takeaways.push({
            icon: 'fa-shield-alt',
            color: 'success',
            text: 'Customer has strong protective factors against churn'
        });
    } else {
        takeaways.push({
            icon: 'fa-balance-scale',
            color: 'warning',
            text: 'Balanced mix of risk and protective factors'
        });
    }
    
    // Top risk factor
    if (increasingFactors.length > 0) {
        const topRisk = increasingFactors[0];
        takeaways.push({
            icon: 'fa-arrow-up',
            color: 'danger',
            text: `Primary risk: ${topRisk.simplified_name}`
        });
    }
    
    // Top protective factor
    if (decreasingFactors.length > 0) {
        const topProtective = decreasingFactors[0];
        takeaways.push({
            icon: 'fa-arrow-down',
            color: 'success',
            text: `Key strength: ${topProtective.simplified_name}`
        });
    }
    
    // Number of significant factors
    const significantFactors = allContributions.filter(c => c.abs_contribution > 0.02).length;
    if (significantFactors > 5) {
        takeaways.push({
            icon: 'fa-chart-line',
            color: 'info',
            text: `Multiple factors (${significantFactors}) significantly influence churn risk`
        });
    }
    
    // Recommendation focus
    if (increasingFactors.length > decreasingFactors.length) {
        takeaways.push({
            icon: 'fa-bullseye',
            color: 'primary',
            text: 'Focus interventions on addressing the top risk factors'
        });
    } else {
        takeaways.push({
            icon: 'fa-bullseye',
            color: 'primary',
            text: 'Leverage existing strengths while addressing key risk areas'
        });
    }
    
    return takeaways.slice(0, 5); // Return top 5 takeaways
}

// Make functions globally available
window.handleFormSubmit = handleFormSubmit;
window.loadSampleData = loadSampleData;
window.showModelInfo = showModelInfo;
window.checkModelStatus = checkModelStatus;

// Helper function to reset form
window.resetForm = function() {
    document.getElementById('resultsCard').style.display = 'none';
    document.getElementById('welcomeCard').style.display = 'block';
    document.getElementById('riskFactorsSummary').style.display = 'none';

    // Clear form
    document.getElementById('age').value = '';
    document.getElementById('frequent_flyer').value = 'No';
    document.getElementById('income_class').value = 'Middle Income';
    document.getElementById('services_opted').value = '';
    document.getElementById('social_sync').value = 'No';
    document.getElementById('booked_hotel').value = 'No';

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
};

// Add keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl+Enter to submit form
    if (event.ctrlKey && event.key === 'Enter') {
        event.preventDefault();
        handleFormSubmit(event);
    }
    // F5 to load sample data
    if (event.key === 'F5') {
        event.preventDefault();
        loadSampleData();
    }
});