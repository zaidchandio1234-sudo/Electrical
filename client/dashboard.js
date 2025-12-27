// EnergyWise Pro Dashboard JavaScript
class Dashboard {
    constructor() {
        this.API_URL = 'http://localhost:8000/api/v1';
        this.authToken = localStorage.getItem('energywise_token');
        this.init();
    }

    async init() {
        this.loadDashboardData();
        this.initCharts();
        this.setupEventListeners();
        this.updateRealTimeData();
        
        // Update real-time data every 10 seconds
        setInterval(() => this.updateRealTimeData(), 10000);
    }

    async loadDashboardData() {
        try {
            // Load multiple data sources in parallel
            const [summary, usage, alerts, devices] = await Promise.all([
                this.fetchDashboardSummary(),
                this.fetchHourlyUsage(1),
                this.fetchAlerts(true),
                this.fetchDevices()
            ]);

            this.updateSummaryCards(summary);
            this.updateUsageChart(usage);
            this.updateAlerts(alerts);
            this.updateDevices(devices);
            
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
            this.showFallbackData();
        }
    }

    async fetchDashboardSummary() {
        const response = await fetch(`${this.API_URL}/dashboard/summary`, {
            headers: {
                'Authorization': `Bearer ${this.authToken}`
            }
        });
        return await response.json();
    }

    async fetchHourlyUsage(days) {
        const response = await fetch(`${this.API_URL}/dashboard/usage/hourly?days=${days}`, {
            headers: {
                'Authorization': `Bearer ${this.authToken}`
            }
        });
        return await response.json();
    }

    async fetchAlerts(unreadOnly = false) {
        const response = await fetch(`${this.API_URL}/alerts?unread_only=${unreadOnly}`, {
            headers: {
                'Authorization': `Bearer ${this.authToken}`
            }
        });
        return await response.json();
    }

    async fetchDevices() {
        const response = await fetch(`${this.API_URL}/devices`, {
            headers: {
                'Authorization': `Bearer ${this.authToken}`
            }
        });
        return await response.json();
    }

    updateSummaryCards(data) {
        // Update real-time card
        document.getElementById('currentUsage').textContent = `${data.real_time.current_usage_kw} kW`;
        
        // Update today's total
        document.getElementById('todayTotal').textContent = `${data.today.total_kwh} kWh`;
        document.getElementById('todayCost').textContent = `$${data.today.cost}`;
        
        // Update month projection
        document.getElementById('monthProjection').textContent = `$${data.month.projected_cost}`;
        document.getElementById('efficiencyScore').textContent = `${data.efficiency.score}/100`;
    }

    initCharts() {
        // Initialize real-time chart
        this.realTimeChart = new Chart(
            document.getElementById('realTimeChart').getContext('2d'),
            {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Power Usage (kW)',
                        data: [],
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
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'kW'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Time'
                            }
                        }
                    }
                }
            }
        );
    }

    updateUsageChart(usageData) {
        const data = usageData.data.slice(-24); // Last 24 hours
        
        this.realTimeChart.data.labels = data.map(d => {
            const date = new Date(d.timestamp);
            return date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        });
        
        this.realTimeChart.data.datasets[0].data = data.map(d => d.usage_kw);
        this.realTimeChart.update();
    }

    updateAlerts(alertsData) {
        const alertsContainer = document.getElementById('alertsList');
        if (!alertsContainer) return;
        
        alertsContainer.innerHTML = '';
        
        alertsData.alerts.slice(0, 3).forEach(alert => {
            const alertElement = document.createElement('div');
            alertElement.className = `alert-item ${alert.severity}`;
            alertElement.innerHTML = `
                <div class="alert-icon">
                    <i class="fas fa-${this.getAlertIcon(alert.type)}"></i>
                </div>
                <div class="alert-content">
                    <h4>${alert.message}</h4>
                    <p>${this.formatTime(alert.timestamp)}</p>
                </div>
                <button class="btn-alert-action" onclick="dashboard.markAlertRead('${alert.id}')">
                    ${alert.action_required ? 'Review' : 'Dismiss'}
                </button>
            `;
            alertsContainer.appendChild(alertElement);
        });
    }

    updateDevices(devicesData) {
        const devicesContainer = document.getElementById('devicesList');
        if (!devicesContainer) return;
        
        devicesContainer.innerHTML = '';
        
        devicesData.devices.forEach(device => {
            const deviceElement = document.createElement('div');
            deviceElement.className = 'device-item';
            deviceElement.innerHTML = `
                <div class="device-icon">
                    <i class="fas fa-${this.getDeviceIcon(device.type)}"></i>
                </div>
                <div class="device-info">
                    <h4>${device.name}</h4>
                    <p>${device.power_usage} kW • ${device.room}</p>
                </div>
                <div class="device-status ${device.status}"></div>
            `;
            devicesContainer.appendChild(deviceElement);
        });
    }

    async updateRealTimeData() {
        try {
            const response = await fetch(`${this.API_URL}/dashboard/summary`, {
                headers: {
                    'Authorization': `Bearer ${this.authToken}`
                }
            });
            
            const data = await response.json();
            
            // Update real-time display
            const realTimeElement = document.getElementById('realTimeUsage');
            if (realTimeElement) {
                realTimeElement.textContent = `${data.real_time.current_usage_kw} kW`;
            }
            
        } catch (error) {
            console.error('Failed to update real-time data:', error);
        }
    }

    getAlertIcon(type) {
        const icons = {
            'usage_spike': 'exclamation-triangle',
            'peak_hour': 'clock',
            'savings_achieved': 'check-circle',
            'device_offline': 'plug'
        };
        return icons[type] || 'bell';
    }

    getDeviceIcon(type) {
        const icons = {
            'thermostat': 'thermometer-half',
            'lighting': 'lightbulb',
            'outlet': 'plug',
            'appliance': 'blender'
        };
        return icons[type] || 'microchip';
    }

    formatTime(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        
        if (diff < 60000) return 'Just now';
        if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
        return date.toLocaleDateString();
    }

    async markAlertRead(alertId) {
        try {
            await fetch(`${this.API_URL}/alerts/${alertId}/read`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.authToken}`
                }
            });
            
            // Reload alerts
            this.loadDashboardData();
            
        } catch (error) {
            console.error('Failed to mark alert as read:', error);
        }
    }

    showFallbackData() {
        // Show sample data if API fails
        const sampleData = {
            real_time: { current_usage_kw: 2.4 },
            today: { total_kwh: 28.5, cost: 4.28 },
            month: { projected_cost: 63.00 },
            efficiency: { score: 78 }
        };
        
        this.updateSummaryCards(sampleData);
        
        // Show error message
        const errorToast = document.createElement('div');
        errorToast.className = 'error-toast';
        errorToast.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <span>Using demo data. API connection issue.</span>
        `;
        document.body.appendChild(errorToast);
        
        setTimeout(() => errorToast.remove(), 5000);
    }

    setupEventListeners() {
        // Date selector buttons
        document.querySelectorAll('.btn-date').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.btn-date').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                
                const days = e.target.textContent === 'Today' ? 1 :
                            e.target.textContent === 'Week' ? 7 :
                            e.target.textContent === 'Month' ? 30 : 365;
                
                this.fetchHourlyUsage(days).then(data => this.updateUsageChart(data));
            });
        });

        // Generate forecast button
        const forecastBtn = document.querySelector('.btn-generate-forecast');
        if (forecastBtn) {
            forecastBtn.addEventListener('click', async () => {
                forecastBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
                forecastBtn.disabled = true;
                
                try {
                    const response = await fetch(`${this.API_URL}/forecast/generate`, {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${this.authToken}`
                        }
                    });
                    
                    const forecast = await response.json();
                    window.location.href = `forecast.html?data=${encodeURIComponent(JSON.stringify(forecast))}`;
                    
                } catch (error) {
                    alert('Failed to generate forecast. Please try again.');
                } finally {
                    forecastBtn.innerHTML = '<i class="fas fa-magic"></i> Generate 7-Day Forecast';
                    forecastBtn.disabled = false;
                }
            });
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});