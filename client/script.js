// EnergyWise Pro - Complete JavaScript with PKR Currency
class EnergyWiseApp {
    constructor() {
        this.API_URL = 'http://localhost:8000';
        this.demoData = null;
        this.charts = {};
        this.usdToPkrRate = 280; // 1 USD = 280 PKR
        this.init();
    }

    init() {
        // Initialize all components
        this.setupEventListeners();
        this.checkAPIStatus();
        this.loadDemoData();
        this.setupCharts();
        this.updateLiveData();
        
        // Start periodic updates
        setInterval(() => this.updateLiveData(), 10000);
        setInterval(() => this.checkAPIStatus(), 30000);
    }

    setupEventListeners() {
        // Mobile menu
        document.getElementById('mobileMenuBtn')?.addEventListener('click', () => {
            document.querySelector('.nav-menu').classList.toggle('active');
        });

        // Live demo button
        document.getElementById('liveDemoBtn')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.runLiveDemo();
        });

        // Generate forecast button
        document.getElementById('generateForecastBtn')?.addEventListener('click', () => {
            this.generateForecast();
        });

        // Run demo button
        document.getElementById('runDemoBtn')?.addEventListener('click', () => {
            this.runSelectedDemo();
        });

        // Export demo button
        document.getElementById('exportDemoBtn')?.addEventListener('click', () => {
            this.exportDemoData();
        });

        // View details button - FIXED: Direct navigation to forecast.html
        document.getElementById('viewDetailsBtn')?.addEventListener('click', () => {
            this.navigateToForecast();
        });

        // Copy code button
        document.getElementById('copyCodeBtn')?.addEventListener('click', () => {
            this.copyCodeSnippet();
        });

        // Demo type change
        document.getElementById('demoType')?.addEventListener('change', () => {
            this.updateDemoUI();
        });
    }

    async checkAPIStatus() {
        try {
            const startTime = performance.now();
            const response = await fetch(`${this.API_URL}/health`);
            const endTime = performance.now();
            
            const responseTime = Math.round(endTime - startTime);
            const data = await response.json();
            
            this.updateAPIStatus(true, responseTime, data);
        } catch (error) {
            this.updateAPIStatus(false, 0, null);
        }
    }

    updateAPIStatus(isOnline, responseTime, data) {
        const indicator = document.getElementById('apiStatusIndicator');
        const responseTimeEl = document.getElementById('apiResponseTime');
        const uptimeEl = document.getElementById('apiUptime');
        const activeUsersEl = document.getElementById('apiActiveUsers');
        
        if (isOnline) {
            indicator.innerHTML = '<span class="status-dot active"></span><span>Online</span>';
            indicator.style.color = '#10b981';
            
            responseTimeEl.textContent = `${responseTime} ms`;
            uptimeEl.textContent = '99.9%';
            activeUsersEl.textContent = data?.active_users || Math.floor(Math.random() * 1000) + 500;
            
            // Update hero visualization status
            const vizStatus = document.querySelector('.viz-status');
            if (vizStatus) {
                vizStatus.innerHTML = '<span class="status-dot active"></span><span>Connected to API</span>';
            }
        } else {
            indicator.innerHTML = '<span class="status-dot inactive"></span><span>Offline</span>';
            indicator.style.color = '#ef4444';
            
            responseTimeEl.textContent = '-- ms';
            uptimeEl.textContent = '-- %';
            activeUsersEl.textContent = '--';
        }
    }

    setupCharts() {
        // Mini usage chart in hero
        const miniCtx = document.getElementById('miniUsageChart');
        if (miniCtx) {
            this.charts.miniUsage = new Chart(miniCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: Array.from({length: 12}, (_, i) => `${i*2}:00`),
                    datasets: [{
                        label: 'Usage (kW)',
                        data: Array.from({length: 12}, () => Math.random() * 2 + 0.5),
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { display: false },
                        x: { display: false }
                    },
                    elements: { point: { radius: 0 } }
                }
            });
        }

        // Demo chart
        const demoCtx = document.getElementById('demoChart');
        if (demoCtx) {
            this.charts.demo = new Chart(demoCtx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    datasets: [{
                        label: 'Energy Usage (kWh)',
                        data: [45, 52, 48, 60, 55, 50, 42],
                        backgroundColor: '#10b981',
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (context) => `${context.parsed.y} kWh (PKR ${(context.parsed.y * 25).toFixed(0)})`
                            }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'kWh'
                            },
                            ticks: {
                                callback: function(value) {
                                    return value + ' kWh';
                                }
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Day'
                            }
                        }
                    }
                }
            });
        }
    }

    async updateLiveData() {
        try {
            const response = await fetch(`${this.API_URL}/health`);
            if (!response.ok) {
                this.showSampleData();
                return;
            }
            
            // Simulate live data updates
            const currentUsage = (Math.random() * 2 + 0.5).toFixed(2);
            const todayCostPkr = (currentUsage * 24 * 25).toFixed(0); // PKR rate: 25 per kWh
            const peakHour = `${Math.floor(Math.random() * 12) + 12}:00`;
            const carbonSaved = (Math.random() * 5 + 10).toFixed(1);
            
            // Update UI elements with PKR
            this.updateElement('currentUsageValue', `${currentUsage} kW`);
            this.updateElement('todayCost', `PKR ${todayCostPkr}`);
            this.updateElement('peakHour', peakHour);
            this.updateElement('carbonSaved', `${carbonSaved} kg`);
            
            // Update trend indicator
            const trend = Math.random() > 0.5 ? 'up' : 'down';
            const trendValue = (Math.random() * 15 + 5).toFixed(1);
            this.updateElement('usageTrend', 
                `<i class="fas fa-arrow-${trend}"></i><span>${trendValue}% from average</span>`);
            
            // Update mini chart
            if (this.charts.miniUsage) {
                const newData = Array.from({length: 12}, () => Math.random() * 2 + 0.5);
                this.charts.miniUsage.data.datasets[0].data = newData;
                this.charts.miniUsage.update();
            }
            
        } catch (error) {
            this.showSampleData();
        }
    }

    showSampleData() {
        // Fallback to sample data in PKR
        this.updateElement('currentUsageValue', '1.85 kW');
        this.updateElement('todayCost', 'PKR 1,110'); // 1.85 * 24 * 25 = 1110
        this.updateElement('peakHour', '14:00');
        this.updateElement('carbonSaved', '12.4 kg');
        this.updateElement('usageTrend', 
            '<i class="fas fa-arrow-down"></i><span>8% below average</span>');
        
        // Update stats
        this.updateElement('avgSavings', '15-30%');
        this.updateElement('realTimeStatus', '24/7');
        this.updateElement('aiAccuracy', '98%');
    }

    async loadDemoData() {
        try {
            const response = await fetch(`${this.API_URL}/generate-forecast`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                this.demoData = await response.json();
                this.updateDemoStats();
            } else {
                this.generateSampleDemoData();
            }
        } catch (error) {
            this.generateSampleDemoData();
        }
    }

    generateSampleDemoData() {
        this.demoData = {
            daily_data: [
                { day_name: 'Monday', total_kwh: 42.5, peak_kwh: 3.2 },
                { day_name: 'Tuesday', total_kwh: 48.7, peak_kwh: 3.8 },
                { day_name: 'Wednesday', total_kwh: 45.2, peak_kwh: 3.5 },
                { day_name: 'Thursday', total_kwh: 52.1, peak_kwh: 4.1 },
                { day_name: 'Friday', total_kwh: 49.8, peak_kwh: 3.9 },
                { day_name: 'Saturday', total_kwh: 38.4, peak_kwh: 2.9 },
                { day_name: 'Sunday', total_kwh: 36.2, peak_kwh: 2.7 }
            ],
            metadata: {
                weekly_total: 312.9,
                average_daily: 44.7
            }
        };
        
        this.updateDemoStats();
    }

    updateDemoStats() {
        if (!this.demoData) return;
        
        const weeklyTotal = this.demoData.metadata?.weekly_total || 
                          this.demoData.daily_data.reduce((sum, day) => sum + day.total_kwh, 0);
        
        const predictedWeeklyEl = document.getElementById('predictedWeekly');
        const potentialSavingsEl = document.getElementById('potentialSavings');
        const aiConfidenceEl = document.getElementById('aiConfidence');
        
        if (predictedWeeklyEl) {
            predictedWeeklyEl.textContent = `${weeklyTotal.toFixed(1)} kWh`;
        }
        
        if (potentialSavingsEl) {
            // PKR calculation: 15% savings at PKR 25/kWh
            const savingsPkr = weeklyTotal * 0.15 * 25;
            potentialSavingsEl.textContent = `PKR ${savingsPkr.toFixed(0)}`;
        }
        
        if (aiConfidenceEl) {
            aiConfidenceEl.textContent = '96%';
        }
        
        // Update demo chart if exists
        if (this.charts.demo && this.demoData.daily_data) {
            const labels = this.demoData.daily_data.map(d => d.day_name.substring(0, 3));
            const data = this.demoData.daily_data.map(d => d.total_kwh);
            
            this.charts.demo.data.labels = labels;
            this.charts.demo.data.datasets[0].data = data;
            
            // Update tooltip to show PKR
            this.charts.demo.options.plugins.tooltip.callbacks.label = (context) => {
                const kwh = context.parsed.y;
                const pkr = kwh * 25;
                return `${kwh.toFixed(1)} kWh (PKR ${pkr.toFixed(0)})`;
            };
            
            this.charts.demo.update();
        }
    }

    async generateForecast() {
        const btn = document.getElementById('generateForecastBtn');
        const originalText = btn.innerHTML;
        
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
        btn.disabled = true;
        
        try {
            // Check if user is authenticated
            const token = localStorage.getItem('energywise_token');
            
            let response;
            if (token) {
                // Use enterprise API
                response = await fetch(`${this.API_URL}/api/v1/forecast/generate`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    }
                });
            } else {
                // Use legacy API
                response = await fetch(`${this.API_URL}/generate-forecast`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
            }
            
            if (response.ok) {
                const forecastData = await response.json();
                this.showForecastResult(forecastData);
                
                // Store for later use
                sessionStorage.setItem('last_forecast', JSON.stringify(forecastData));
                
            } else {
                throw new Error('Forecast generation failed');
            }
            
        } catch (error) {
            console.error('Forecast error:', error);
            this.showNotification('Failed to generate forecast. Using sample data.', 'error');
            this.generateSampleDemoData();
        } finally {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }

    showForecastResult(forecastData) {
        this.showNotification('7-day forecast generated successfully!', 'success');
        
        // Update demo data with new forecast
        this.demoData = forecastData;
        this.updateDemoStats();
        
        // Store data for forecast page
        sessionStorage.setItem('last_forecast', JSON.stringify(forecastData));
        
        // Ask user if they want to view details
        setTimeout(() => {
            const viewDetails = confirm('Forecast generated successfully! Would you like to view the detailed analysis?');
            if (viewDetails) {
                this.navigateToForecast();
            }
        }, 500);
    }

    // NEW METHOD: Navigate to forecast page
    navigateToForecast() {
        // Store data in session storage for forecast page
        if (this.demoData) {
            sessionStorage.setItem('last_forecast', JSON.stringify(this.demoData));
        }
        
        // Navigate to forecast page
        window.location.href = 'forecast.html';
    }

    async runLiveDemo() {
        const btn = document.getElementById('liveDemoBtn');
        const originalText = btn.innerHTML;
        
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running Demo...';
        btn.disabled = true;
        
        // Show loading animation
        this.showLoadingAnimation();
        
        try {
            // Simulate API call
            await new Promise(resolve => setTimeout(resolve, 1500));
            
            // Generate sample forecast
            await this.generateForecast();
            
            // Scroll to demo section
            document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
            
            // Update demo with latest data
            this.runSelectedDemo();
            
        } finally {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }

    runSelectedDemo() {
        const demoType = document.getElementById('demoType').value;
        const dataSource = document.getElementById('dataSource').value;
        
        this.showNotification(`Running ${demoType} demo with ${dataSource} data...`, 'info');
        
        // Update chart based on demo type
        if (this.charts.demo) {
            let labels, data, chartType, backgroundColor;
            
            switch (demoType) {
                case 'forecast':
                    labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
                    data = [42.5, 48.7, 45.2, 52.1, 49.8, 38.4, 36.2];
                    chartType = 'bar';
                    backgroundColor = '#10b981';
                    break;
                    
                case 'realtime':
                    labels = Array.from({length: 24}, (_, i) => `${i}:00`);
                    data = Array.from({length: 24}, (_, i) => {
                        const base = 1.5 + 0.5 * Math.sin(i/24 * 2 * Math.PI);
                        return base + (Math.random() * 0.4 - 0.2);
                    });
                    chartType = 'line';
                    backgroundColor = 'rgba(59, 130, 246, 0.1)';
                    break;
                    
                case 'savings':
                    labels = ['Week 1', 'Week 2', 'Week 3', 'Week 4'];
                    data = [1520, 1850, 2230, 2570]; // PKR amounts
                    chartType = 'bar';
                    backgroundColor = '#f59e0b';
                    break;
                    
                case 'analytics':
                    labels = ['AC', 'Lighting', 'Appliances', 'Electronics', 'Other'];
                    data = [35, 20, 25, 15, 5];
                    chartType = 'pie';
                    backgroundColor = ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6', '#ef4444'];
                    break;
            }
            
            // Update chart
            this.charts.demo.data.labels = labels;
            this.charts.demo.data.datasets[0].data = data;
            this.charts.demo.data.datasets[0].backgroundColor = backgroundColor;
            this.charts.demo.config.type = chartType;
            
            // Update tooltip for PKR
            if (demoType === 'forecast') {
                this.charts.demo.options.plugins.tooltip.callbacks.label = (context) => {
                    const kwh = context.parsed.y;
                    const pkr = kwh * 25;
                    return `${kwh.toFixed(1)} kWh (PKR ${pkr.toFixed(0)})`;
                };
            } else if (demoType === 'savings') {
                this.charts.demo.options.plugins.tooltip.callbacks.label = (context) => {
                    return `PKR ${context.parsed.y.toFixed(0)}`;
                };
            }
            
            this.charts.demo.update();
            
            // Update stats
            this.updateDemoStats();
        }
    }

    updateDemoUI() {
        const demoType = document.getElementById('demoType').value;
        const demoStats = document.querySelector('.demo-stats');
        
        if (demoStats) {
            switch (demoType) {
                case 'forecast':
                    demoStats.innerHTML = `
                        <div class="demo-stat">
                            <div class="demo-stat-label">Predicted Weekly Usage</div>
                            <div class="demo-stat-value" id="predictedWeekly">-- kWh</div>
                        </div>
                        <div class="demo-stat">
                            <div class="demo-stat-label">Potential Savings</div>
                            <div class="demo-stat-value" id="potentialSavings">PKR --</div>
                        </div>
                        <div class="demo-stat">
                            <div class="demo-stat-label">AI Confidence</div>
                            <div class="demo-stat-value" id="aiConfidence">-- %</div>
                        </div>
                    `;
                    break;
                    
                case 'realtime':
                    demoStats.innerHTML = `
                        <div class="demo-stat">
                            <div class="demo-stat-label">Current Usage</div>
                            <div class="demo-stat-value" id="currentUsage">-- kW</div>
                        </div>
                        <div class="demo-stat">
                            <div class="demo-stat-label">Today's Cost</div>
                            <div class="demo-stat-value" id="todayCostDemo">PKR --</div>
                        </div>
                        <div class="demo-stat">
                            <div class="demo-stat-label">Peak Load</div>
                            <div class="demo-stat-value" id="peakLoad">-- kW</div>
                        </div>
                    `;
                    break;
                    
                case 'savings':
                    demoStats.innerHTML = `
                        <div class="demo-stat">
                            <div class="demo-stat-label">Monthly Savings</div>
                            <div class="demo-stat-value" id="monthlySavings">PKR --</div>
                        </div>
                        <div class="demo-stat">
                            <div class="demo-stat-label">Yearly Projection</div>
                            <div class="demo-stat-value" id="yearlySavings">PKR --</div>
                        </div>
                        <div class="demo-stat">
                            <div class="demo-stat-label">ROI</div>
                            <div class="demo-stat-value" id="roiPercentage">-- %</div>
                        </div>
                    `;
                    break;
                    
                case 'analytics':
                    demoStats.innerHTML = `
                        <div class="demo-stat">
                            <div class="demo-stat-label">Total Usage</div>
                            <div class="demo-stat-value" id="totalUsage">-- kWh</div>
                        </div>
                        <div class="demo-stat">
                            <div class="demo-stat-label">Efficiency Score</div>
                            <div class="demo-stat-value" id="efficiencyScore">--/100</div>
                        </div>
                        <div class="demo-stat">
                            <div class="demo-stat-label">Carbon Footprint</div>
                            <div class="demo-stat-value" id="carbonFootprint">-- kg</div>
                        </div>
                    `;
                    break;
            }
            
            // Update the stats
            this.updateDemoStats();
        }
    }

    exportDemoData() {
        if (!this.demoData) {
            this.showNotification('No data available to export', 'warning');
            return;
        }
        
        // Create CSV content with PKR
        let csvContent = 'Day,Usage (kWh),Peak (kW),Cost (PKR)\n';
        
        if (this.demoData.daily_data) {
            this.demoData.daily_data.forEach(day => {
                const costPkr = (day.total_kwh * 25).toFixed(0);
                csvContent += `${day.day_name},${day.total_kwh},${day.peak_kwh || 'N/A'},PKR ${costPkr}\n`;
            });
        }
        
        // Create download link
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `energywise-forecast-${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        this.showNotification('Data exported successfully!', 'success');
    }

    copyCodeSnippet() {
        const codeElement = document.getElementById('apiCodeSnippet');
        const code = codeElement.textContent;
        
        navigator.clipboard.writeText(code).then(() => {
            const btn = document.getElementById('copyCodeBtn');
            const originalText = btn.innerHTML;
            
            btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            btn.style.backgroundColor = '#10b981';
            
            setTimeout(() => {
                btn.innerHTML = originalText;
                btn.style.backgroundColor = '';
            }, 2000);
            
            this.showNotification('Code copied to clipboard!', 'success');
        }).catch(err => {
            console.error('Failed to copy code:', err);
            this.showNotification('Failed to copy code', 'error');
        });
    }

    // Utility methods
    updateElement(id, content) {
        const element = document.getElementById(id);
        if (element) {
            element.innerHTML = content;
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${this.getNotificationIcon(type)}"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    getNotificationIcon(type) {
        switch (type) {
            case 'success': return 'check-circle';
            case 'error': return 'exclamation-circle';
            case 'warning': return 'exclamation-triangle';
            default: return 'info-circle';
        }
    }

    showLoadingAnimation() {
        // Create loading overlay
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-spinner">
                <div class="spinner"></div>
                <p>Processing your request...</p>
            </div>
        `;
        
        document.body.appendChild(overlay);
        
        // Auto-remove after 2 seconds
        setTimeout(() => {
            if (overlay.parentElement) {
                overlay.remove();
            }
        }, 2000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.energyWiseApp = new EnergyWiseApp();
    
    // Check if user is already logged in
    const token = localStorage.getItem('energywise_token');
    if (token && window.location.pathname.includes('index.html')) {
        // Update CTA buttons for logged-in users
        const ctaButtons = document.querySelector('.cta-actions');
        if (ctaButtons) {
            ctaButtons.innerHTML = `
                <a href="dashboard.html" class="btn btn-primary btn-xlarge">
                    <i class="fas fa-tachometer-alt"></i>
                    Go to Dashboard
                </a>
                <a href="forecast.html" class="btn btn-outline btn-xlarge">
                    <i class="fas fa-chart-line"></i>
                    View Forecasts
                </a>
            `;
        }
    }
});

// Global helper functions with PKR
function formatNumber(num) {
    return new Intl.NumberFormat('en-PK').format(num);
}

function formatCurrencyPKR(amount) {
    return new Intl.NumberFormat('en-PK', {
        style: 'currency',
        currency: 'PKR',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

function formatCurrency(amount, currency = 'PKR') {
    if (currency === 'PKR') {
        return formatCurrencyPKR(amount);
    } else {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(amount);
    }
}

function convertUSDtoPKR(usdAmount) {
    return usdAmount * 280; // 1 USD = 280 PKR
}

function getCurrentTime() {
    return new Date().toLocaleTimeString('en-PK', { hour: '2-digit', minute: '2-digit' });
}

// Add CSS for notifications and loading
const style = document.createElement('style');
style.textContent = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 20px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        z-index: 10000;
        animation: slideIn 0.3s ease;
        max-width: 400px;
    }
    
    .notification-success {
        background: #10b981;
        color: white;
        border-left: 4px solid #059669;
    }
    
    .notification-error {
        background: #ef4444;
        color: white;
        border-left: 4px solid #dc2626;
    }
    
    .notification-warning {
        background: #f59e0b;
        color: white;
        border-left: 4px solid #d97706;
    }
    
    .notification-info {
        background: #3b82f6;
        color: white;
        border-left: 4px solid #2563eb;
    }
    
    .notification button {
        background: none;
        border: none;
        color: inherit;
        cursor: pointer;
        margin-left: auto;
        opacity: 0.8;
    }
    
    .notification button:hover {
        opacity: 1;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.9);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    }
    
    .loading-spinner {
        text-align: center;
    }
    
    .loading-spinner .spinner {
        width: 60px;
        height: 60px;
        border: 4px solid #e5e7eb;
        border-top-color: #10b981;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 20px;
    }
    
    .loading-spinner p {
        color: #6b7280;
        font-weight: 500;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-dot.active {
        background: #10b981;
        animation: pulse 2s infinite;
    }
    
    .status-dot.inactive {
        background: #ef4444;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .btn-viz-action {
        width: 100%;
        padding: 12px;
        background: linear-gradient(135deg, #10b981, #3b82f6);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        transition: all 0.3s ease;
    }
    
    .btn-viz-action:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(16, 185, 129, 0.2);
    }
    
    .nav-menu.active {
        display: flex !important;
        flex-direction: column;
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: white;
        padding: 20px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
`;
document.head.appendChild(style);