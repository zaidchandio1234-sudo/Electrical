
// Complete Forecast System with Error Display
class CompleteForecast {
    constructor() {
        this.API_URL = 'http://localhost:8000';
        this.data = null;
        this.chart = null;
        this.init();
    }

    async init() {
        console.log("🚀 Starting Energy Forecast System...");
        
        // Show initial loading
        this.showLoading("Initializing forecast system...");
        
        try {
            // Check API connection first
            const isConnected = await this.checkAPIConnection();
            
            if (isConnected) {
                await this.loadForecast();
                this.setupEventListeners();
            } else {
                this.showConnectionError();
                this.loadSampleData();
            }
        } catch (error) {
            console.error("Initialization error:", error);
            this.showError("System initialization failed", error.message);
            this.loadSampleData();
        } finally {
            this.hideLoading();
        }
    }

    async checkAPIConnection() {
        try {
            console.log("🔌 Checking API connection...");
            
            const response = await fetch(`${this.API_URL}/health`, {
                method: 'GET',
                timeout: 5000
            });
            
            if (response.ok) {
                const data = await response.json();
                console.log("✅ API is healthy:", data);
                return true;
            }
            return false;
        } catch (error) {
            console.log("❌ API check failed:", error.message);
            return false;
        }
    }

    async loadForecast() {
        console.log("📡 Loading forecast data...");
        
        // Show loading message
        this.showLoading("Fetching AI predictions...");
        
        try {
            const response = await fetch(`${this.API_URL}/api/v1/forecast/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    days: 7,
                    include_hourly: true,
                    include_recommendations: true
                })
            });

            console.log("Response status:", response.status);
            
            if (response.ok) {
                this.data = await response.json();
                console.log("✅ Forecast data received:", this.data);
                
                // Save to localStorage for offline use
                localStorage.setItem('last_forecast', JSON.stringify(this.data));
                localStorage.setItem('forecast_time', new Date().toISOString());
                
                this.render();
                this.showSuccess("AI forecast loaded successfully!");
                
            } else {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
        } catch (error) {
            console.error("❌ Forecast loading failed:", error);
            
            // Try legacy endpoint
            await this.tryLegacyEndpoint();
        }
    }

    async tryLegacyEndpoint() {
        console.log("🔄 Trying legacy endpoint...");
        
        try {
            const response = await fetch(`${this.API_URL}/generate-forecast`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                this.data = await response.json();
                console.log("✅ Legacy endpoint success");
                this.render();
                this.showWarning("Using legacy API endpoint");
            } else {
                throw new Error("Legacy endpoint failed");
            }
            
        } catch (error) {
            console.log("❌ All API endpoints failed");
            throw error;
        }
    }

    loadSampleData() {
        console.log("📊 Loading sample data...");
        
        this.showWarning(
            "API Connection Failed", 
            "Using sample data. Please check if the API server is running.<br><br>" +
            "Run this in terminal:<br>" +
            "<code>cd server && python api.py</code>"
        );
        
        const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
        const currentDate = new Date();
        
        this.data = {
            daily_data: days.map((dayName, index) => {
                const date = new Date(currentDate);
                date.setDate(date.getDate() + index);
                
                const totalKwh = 40 + Math.random() * 15;
                const costPkr = totalKwh * 50;
                
                return {
                    day_name: dayName,
                    date: date.toISOString().split('T')[0],
                    total_kwh: parseFloat(totalKwh.toFixed(1)),
                    peak_kwh: parseFloat((3 + Math.random() * 1.5).toFixed(1)),
                    cost_pkr: Math.round(costPkr),
                    savings_potential_pkr: Math.round(costPkr * 0.15),
                    is_peak: index === 2 || index === 3 // Wed/Thu are peaks
                };
            }),
            metadata: {
                weekly_total: 312.9,
                weekly_cost_pkr: 15645,
                weekly_savings_potential_pkr: 2550,
                average_daily: 44.7,
                peak_day: 'Thursday',
                peak_usage: 52.1,
                confidence: 85,
                generated_at: new Date().toISOString(),
                carbon_savings_kg: 1250,
                note: 'SAMPLE DATA - API not connected'
            },
            recommendations: [
                {
                    title: 'Optimize AC Schedule',
                    description: 'Set temperature to 24°C during peak hours (2 PM - 4 PM)',
                    impact: 'high',
                    savings_pkr: 500
                },
                {
                    title: 'Shift Laundry Schedule',
                    description: 'Move laundry to after 8 PM to avoid peak hour charges',
                    impact: 'medium',
                    savings_pkr: 300
                },
                {
                    title: 'EV Charging Optimization',
                    description: 'Charge electric vehicle between 12 AM - 6 AM for maximum savings',
                    impact: 'high',
                    savings_pkr: 800
                },
                {
                    title: 'Solar Pre-cooling',
                    description: 'Use solar power to pre-cool home before peak hours begin',
                    impact: 'medium',
                    savings_pkr: 400
                }
            ],
            insights: {
                summary: 'Sample data - Connect to API for live predictions',
                trend: 'stable',
                efficiency_score: 75
            }
        };
        
        this.render();
    }

    render() {
        if (!this.data) {
            this.showError("No data available", "Failed to load forecast data");
            return;
        }
        
        console.log("🎨 Rendering forecast data...");
        
        this.updateStats();
        this.updateChart();
        this.updateDays();
        this.updateRecommendations();
        this.updateSavings();
        this.updateTimestamp();
        
        // Add animations
        this.animateElements();
    }

    updateStats() {
        if (!this.data.metadata) return;
        
        const meta = this.data.metadata;
        const statsContainer = document.getElementById('forecastStats');
        if (!statsContainer) return;
        
        statsContainer.innerHTML = `
            <div class="stat-card">
                <div class="stat-header">
                    <div class="stat-icon">
                        <i class="fas fa-bolt"></i>
                    </div>
                    <span class="stat-title">Weekly Total</span>
                </div>
                <div class="stat-value">${meta.weekly_total?.toFixed(1) || '0.0'}<span class="stat-unit">kWh</span></div>
                <div class="stat-trend trend-down">
                    <i class="fas fa-arrow-down"></i>
                    <span>-8% from last week</span>
                </div>
            </div>
            
            <div class="stat-card">
                <div class="stat-header">
                    <div class="stat-icon">
                        <i class="fas fa-dollar-sign"></i>
                    </div>
                    <span class="stat-title">Projected Cost</span>
                </div>
                <div class="stat-value">PKR ${this.formatNumber(meta.weekly_cost_pkr || 0)}</div>
                <div class="stat-trend trend-down">
                    <i class="fas fa-arrow-down"></i>
                    <span>Save PKR ${this.formatNumber(meta.weekly_savings_potential_pkr || 0)}</span>
                </div>
            </div>
            
            <div class="stat-card">
                <div class="stat-header">
                    <div class="stat-icon">
                        <i class="fas fa-robot"></i>
                    </div>
                    <span class="stat-title">AI Confidence</span>
                </div>
                <div class="stat-value">${meta.confidence || 0}<span class="stat-unit">%</span></div>
                <div class="stat-trend trend-up">
                    <i class="fas fa-arrow-up"></i>
                    <span>${meta.confidence > 90 ? 'High' : 'Medium'} accuracy</span>
                </div>
            </div>
            
            <div class="stat-card">
                <div class="stat-header">
                    <div class="stat-icon">
                        <i class="fas fa-leaf"></i>
                    </div>
                    <span class="stat-title">Carbon Impact</span>
                </div>
                <div class="stat-value">-${meta.carbon_savings_kg || 0}<span class="stat-unit">kg</span></div>
                <div class="stat-trend trend-up">
                    <i class="fas fa-arrow-up"></i>
                    <span>Reduced footprint</span>
                </div>
            </div>
        `;
    }

    updateChart() {
        const canvas = document.getElementById('forecastChart');
        if (!canvas || !this.data.daily_data) return;
        
        const ctx = canvas.getContext('2d');
        const dailyData = this.data.daily_data;
        const peakDay = this.data.metadata?.peak_day || 'Thursday';
        
        // Destroy existing chart
        if (this.chart) {
            this.chart.destroy();
        }
        
        this.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: dailyData.map(d => d.day_name.substring(0, 3)),
                datasets: [{
                    label: 'Energy Usage (kWh)',
                    data: dailyData.map(d => d.total_kwh),
                    backgroundColor: dailyData.map(d => 
                        d.day_name === peakDay ? 'rgba(255, 149, 0, 0.8)' : 'rgba(0, 122, 255, 0.7)'
                    ),
                    borderColor: dailyData.map(d => 
                        d.day_name === peakDay ? 'rgba(255, 149, 0, 1)' : 'rgba(0, 122, 255, 1)'
                    ),
                    borderWidth: 1,
                    borderRadius: 6,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.9)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 8,
                        displayColors: false,
                        callbacks: {
                            label: (context) => {
                                const day = dailyData[context.dataIndex];
                                return [
                                    `Usage: ${context.parsed.y.toFixed(1)} kWh`,
                                    `Cost: PKR ${this.formatNumber(day.cost_pkr || 0)}`,
                                    `Save: PKR ${this.formatNumber(day.savings_potential_pkr || 0)}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.5)',
                            callback: (value) => value + ' kWh'
                        }
                    },
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'rgba(255, 255, 255, 0.5)' }
                    }
                }
            }
        });
    }

    updateDays() {
        const container = document.querySelector('.forecast-days');
        if (!container || !this.data.daily_data) return;
        
        const dailyData = this.data.daily_data;
        const peakDay = this.data.metadata?.peak_day || 'Thursday';
        
        container.innerHTML = dailyData.map((day, index) => {
            const date = new Date(day.date);
            const dayNum = date.getDate();
            const isPeak = day.day_name === peakDay || day.is_peak;
            
            return `
                <div class="day-card ${isPeak ? 'peak' : ''}" data-index="${index}">
                    <div class="day-header">
                        <span class="day-name">${day.day_name.substring(0, 3)}</span>
                        ${isPeak 
                            ? '<span class="peak-badge"><i class="fas fa-bolt"></i> PEAK</span>' 
                            : `<span class="day-date">${dayNum}</span>`
                        }
                    </div>
                    <div class="day-metrics">
                        <div class="metric">
                            <div class="metric-value">${day.total_kwh.toFixed(1)}</div>
                            <div class="metric-label">kWh</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${day.peak_kwh?.toFixed(1) || '3.5'}</div>
                            <div class="metric-label">Peak</div>
                        </div>
                    </div>
                    <div class="stat-trend ${isPeak ? 'trend-up' : 'trend-down'}">
                        <i class="fas fa-arrow-${isPeak ? 'up' : 'down'}"></i>
                        <span>${isPeak ? 'High usage' : 'Normal day'}</span>
                    </div>
                </div>
            `;
        }).join('');
        
        // Add click handlers
        this.setupDayCardClicks();
    }

    updateRecommendations() {
        const container = document.querySelector('.recommendations-grid');
        if (!container || !this.data.recommendations) return;
        
        const recommendations = this.data.recommendations.slice(0, 4);
        
        container.innerHTML = recommendations.map(rec => `
            <div class="recommendation-card">
                <div class="rec-icon">
                    <i class="fas ${this.getRecIcon(rec.title)}"></i>
                </div>
                <div class="rec-content">
                    <h4>${rec.title}</h4>
                    <p>${rec.description}</p>
                    <div class="rec-footer" style="margin-top: 8px; font-size: 12px; color: #34C759; font-weight: 600;">
                        Save PKR ${this.formatNumber(rec.savings_pkr)}/week
                    </div>
                </div>
            </div>
        `).join('');
    }

    updateSavings() {
        const container = document.querySelector('.savings-grid');
        if (!container || !this.data.metadata) return;
        
        const meta = this.data.metadata;
        const weeklySavings = meta.weekly_savings_potential_pkr || 0;
        const monthlySavings = weeklySavings * 4;
        const annualSavings = weeklySavings * 52;
        
        container.innerHTML = `
            <div class="savings-item">
                <div class="savings-label">Weekly Savings</div>
                <div class="savings-amount">PKR ${this.formatNumber(weeklySavings)}</div>
                <div style="font-size: 11px; margin-top: 4px; opacity: 0.7;">
                    Based on AI advice
                </div>
            </div>
            
            <div class="savings-item">
                <div class="savings-label">Monthly Projection</div>
                <div class="savings-amount">PKR ${this.formatNumber(monthlySavings)}</div>
                <div style="font-size: 11px; margin-top: 4px; opacity: 0.7;">
                    4-week estimate
                </div>
            </div>
            
            <div class="savings-item">
                <div class="savings-label">Annual Savings</div>
                <div class="savings-amount">PKR ${this.formatNumber(annualSavings)}</div>
                <div style="font-size: 11px; margin-top: 4px; opacity: 0.7;">
                    Potential yearly savings
                </div>
            </div>
            
            <div class="savings-item">
                <div class="savings-label">Carbon Reduction</div>
                <div class="savings-amount">${meta.carbon_savings_kg || 0} kg CO₂</div>
                <div style="font-size: 11px; margin-top: 4px; opacity: 0.7;">
                    Environmental impact
                </div>
            </div>
        `;
    }

    updateTimestamp() {
        const element = document.getElementById('timestamp');
        if (element && this.data.metadata) {
            const date = new Date(this.data.metadata.generated_at || new Date());
            element.innerHTML = `
                <i class="fas fa-clock"></i>
                Last updated: ${date.toLocaleTimeString([], { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                })}
                ${this.data.metadata.note ? `<br><small style="color: #FF9500;">${this.data.metadata.note}</small>` : ''}
            `;
        }
    }

    setupEventListeners() {
        // Refresh button
        const refreshBtn = document.querySelector('[onclick*="regenerate"]') || 
                          document.querySelector('.export-btn:nth-child(4)');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.regenerateForecast();
            });
        }
        
        // Export buttons
        document.querySelectorAll('.export-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                if (btn.textContent.includes('PDF')) {
                    e.preventDefault();
                    this.exportData('pdf');
                } else if (btn.textContent.includes('CSV')) {
                    e.preventDefault();
                    this.exportData('csv');
                } else if (btn.textContent.includes('Dashboard')) {
                    // Let default behavior (href) work
                }
            });
        });
    }

    setupDayCardClicks() {
        document.querySelectorAll('.day-card').forEach(card => {
            card.addEventListener('click', () => {
                const index = parseInt(card.dataset.index);
                if (this.data.daily_data[index]) {
                    const day = this.data.daily_data[index];
                    this.showDayDetails(day);
                }
            });
        });
    }

    async regenerateForecast() {
        if (confirm('Generate new forecast with latest data?')) {
            // Clear cache
            localStorage.removeItem('last_forecast');
            localStorage.removeItem('forecast_time');
            
            // Reload
            await this.loadForecast();
            
            this.showSuccess("Forecast updated successfully!");
        }
    }

    showDayDetails(day) {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            z-index: 10000;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        `;
        
        modal.innerHTML = `
            <div style="
                background: #1c1c1e;
                border-radius: 16px;
                padding: 24px;
                max-width: 400px;
                width: 100%;
                border: 1px solid rgba(84, 84, 88, 0.65);
                color: white;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h3 style="margin: 0; display: flex; align-items: center; gap: 10px;">
                        <i class="fas fa-calendar-day"></i>
                        ${day.day_name}
                    </h3>
                    <button onclick="this.parentElement.parentElement.parentElement.remove()" style="
                        background: none;
                        border: none;
                        color: white;
                        font-size: 24px;
                        cursor: pointer;
                        padding: 0;
                        width: 40px;
                        height: 40px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">&times;</button>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                        <div>
                            <div style="font-size: 12px; color: rgba(255, 255, 255, 0.5); margin-bottom: 4px;">Total Usage</div>
                            <div style="font-size: 24px; font-weight: 700;">${day.total_kwh} kWh</div>
                        </div>
                        <div>
                            <div style="font-size: 12px; color: rgba(255, 255, 255, 0.5); margin-bottom: 4px;">Peak Load</div>
                            <div style="font-size: 24px; font-weight: 700;">${day.peak_kwh} kW</div>
                        </div>
                    </div>
                    
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 12px; margin-top: 15px;">
                        <div style="font-size: 14px; color: rgba(255, 255, 255, 0.7); margin-bottom: 8px;">Cost Analysis</div>
                        <div style="display: flex; justify-content: space-between;">
                            <span>Estimated Cost:</span>
                            <span style="font-weight: 600; color: #34C759;">PKR ${this.formatNumber(day.cost_pkr)}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                            <span>Savings Potential:</span>
                            <span style="font-weight: 600; color: #FF9500;">PKR ${this.formatNumber(day.savings_potential_pkr)}</span>
                        </div>
                    </div>
                </div>
                
                <button onclick="this.parentElement.parentElement.remove()" style="
                    width: 100%;
                    padding: 12px;
                    background: #007AFF;
                    border: none;
                    border-radius: 12px;
                    color: white;
                    font-weight: 600;
                    cursor: pointer;
                    margin-top: 20px;
                ">
                    Close
                </button>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close on click outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    exportData(format) {
        if (!this.data) {
            this.showError("No data to export", "Load forecast data first");
            return;
        }
        
        if (format === 'csv') {
            this.exportCSV();
        } else {
            this.showInfo("PDF export feature coming soon!");
        }
    }

    exportCSV() {
        let csv = 'Day,Date,Total kWh,Peak kW,Cost (PKR),Savings (PKR)\n';
        
        this.data.daily_data.forEach(day => {
            csv += `${day.day_name},${day.date},${day.total_kwh},${day.peak_kwh},${day.cost_pkr},${day.savings_potential_pkr}\n`;
        });
        
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `energy-forecast-${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        this.showSuccess("CSV exported successfully!");
    }

    // ===== NOTIFICATION SYSTEM =====
    
    showLoading(message = "Loading...") {
        // Create or update loading overlay
        let overlay = document.getElementById('loadingOverlay');
        
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'loadingOverlay';
            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.9);
                backdrop-filter: blur(10px);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                z-index: 9999;
                color: white;
                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            `;
            
            overlay.innerHTML = `
                <div style="
                    width: 50px;
                    height: 50px;
                    border: 3px solid rgba(255, 255, 255, 0.1);
                    border-top-color: #007AFF;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin-bottom: 20px;
                "></div>
                <div id="loadingMessage" style="font-size: 16px; text-align: center;"></div>
                <div style="font-size: 14px; opacity: 0.7; margin-top: 10px;">
                    <i class="fas fa-robot"></i> AI Forecast System
                </div>
            `;
            
            document.body.appendChild(overlay);
            
            // Add spin animation
            const style = document.createElement('style');
            style.textContent = `
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
            `;
            document.head.appendChild(style);
        }
        
        document.getElementById('loadingMessage').textContent = message;
        overlay.style.display = 'flex';
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }

    showError(title, message = "") {
        this.showNotification(title, message, 'error');
    }

    showWarning(title, message = "") {
        this.showNotification(title, message, 'warning');
    }

    showSuccess(message) {
        this.showNotification("Success", message, 'success');
    }

    showInfo(message) {
        this.showNotification("Info", message, 'info');
    }

    showNotification(title, message, type = 'info') {
        // Remove existing notifications
        document.querySelectorAll('.notification-alert').forEach(el => el.remove());
        
        const colors = {
            error: { bg: '#FF3B30', icon: 'fa-exclamation-circle' },
            warning: { bg: '#FF9500', icon: 'fa-exclamation-triangle' },
            success: { bg: '#34C759', icon: 'fa-check-circle' },
            info: { bg: '#007AFF', icon: 'fa-info-circle' }
        };
        
        const color = colors[type] || colors.info;
        
        const notification = document.createElement('div');
        notification.className = 'notification-alert';
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${color.bg};
            color: white;
            padding: 16px 20px;
            border-radius: 12px;
            z-index: 10000;
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
            animation: slideIn 0.3s ease-out;
            max-width: 400px;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        `;
        
        notification.innerHTML = `
            <div style="display: flex; align-items: flex-start; gap: 12px;">
                <i class="fas ${color.icon}" style="font-size: 20px; flex-shrink: 0;"></i>
                <div style="flex: 1;">
                    <div style="font-weight: 600; margin-bottom: ${message ? '4px' : '0'}; font-size: 15px;">
                        ${title}
                    </div>
                    ${message ? `<div style="font-size: 14px; opacity: 0.9; line-height: 1.4;">${message}</div>` : ''}
                </div>
                <button onclick="this.parentElement.parentElement.remove()" style="
                    background: none;
                    border: none;
                    color: white;
                    cursor: pointer;
                    padding: 0;
                    width: 24px;
                    height: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    opacity: 0.7;
                    transition: opacity 0.2s;
                " onmouseover="this.style.opacity='1'" onmouseout="this.style.opacity='0.7'">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
        
        // Add animation
        const style = document.createElement('style');
        if (!document.querySelector('#notification-animations')) {
            style.id = 'notification-animations';
            style.textContent = `
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
            `;
            document.head.appendChild(style);
        }
    }

    showConnectionError() {
        this.showError(
            "API Connection Failed",
            "Cannot connect to forecast server.<br><br>" +
            "Please make sure:<br>" +
            "1. API server is running<br>" +
            "2. Run: <code>python api.py</code> in server folder<br>" +
            "3. Check console for errors"
        );
    }

    // ===== HELPER METHODS =====
    
    getRecIcon(title) {
        if (title.includes('AC') || title.includes('thermometer')) return 'fa-thermometer-half';
        if (title.includes('Laundry') || title.includes('washing')) return 'fa-washing-machine';
        if (title.includes('EV') || title.includes('car')) return 'fa-car';
        if (title.includes('Solar')) return 'fa-sun';
        return 'fa-lightbulb';
    }

    formatNumber(num) {
        return new Intl.NumberFormat('en-PK').format(Math.round(num));
    }

    animateElements() {
        // Add subtle animations to elements
        const elements = document.querySelectorAll('.stat-card, .day-card, .recommendation-card, .savings-item');
        elements.forEach((el, i) => {
            el.style.animation = `fadeInUp 0.5s ease-out ${i * 0.05}s both`;
        });
        
        // Add animation if not exists
        if (!document.querySelector('#fade-animation')) {
            const style = document.createElement('style');
            style.id = 'fade-animation';
            style.textContent = `
                @keyframes fadeInUp {
                    from {
                        opacity: 0;
                        transform: translateY(20px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log("📄 DOM loaded, starting forecast system...");
    window.forecastSystem = new CompleteForecast();
});

// Make helper functions globally available
window.formatNumber = (num) => new Intl.NumberFormat('en-PK').format(num);
