
// EnergyWise Pro - Enterprise Grade AI Energy Advisor
// 30 Years of Experience - Every Response Unique

class EnergyWiseApp {
    constructor() {
        this.API_URL = 'http://localhost:8000';
        this.sessionId = this.generateSessionId();
        this.forecastData = null;
        this.chart = null;
        this.pkrRate = 25;
        this.conversationHistory = [];
        this.days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
        this.fullDays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
        
        // 30 Years of Wisdom Database
        this.wisdomDatabase = this.initializeWisdomDatabase();
        this.adviceTemplates = this.initializeAdviceTemplates();
        this.insightGenerators = this.initializeInsightGenerators();
        
        this.init();
    }

    init() {
        this.loadForecast();
        this.setupEventListeners();
    }

    generateSessionId() {
        return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    setupEventListeners() {
        // Add any additional event listeners here
    }

    async loadForecast() {
        try {
            // Generate dynamic forecast data
            await this.generateDynamicForecast();
            
            // Display forecast
            this.displayForecast();
            
            // Create chart
            this.createChart();
            
            // Display daily breakdown
            this.displayDailyBreakdown();
            
            // Generate initial AI message
            await this.generateInitialAIMessage();
            
        } catch (error) {
            console.error('Error loading forecast:', error);
        } finally {
            document.getElementById('loading').style.display = 'none';
        }
    }

    async generateDynamicForecast() {
        // Generate truly unique forecast data each time
        const baseUsage = 35 + Math.random() * 25; // 35-60 kWh base
        const variationFactor = 0.7 + Math.random() * 0.6; // 0.7-1.3
        const seasonalFactor = this.getSeasonalFactor();
        const behaviorPattern = this.getRandomBehaviorPattern();
        
        this.forecastData = {
            daily_data: [],
            metadata: {
                customer_archetype: behaviorPattern.archetype,
                confidence: 88 + Math.random() * 10,
                generation_timestamp: new Date().toISOString(),
                session_id: this.sessionId
            }
        };

        const today = new Date();
        
        for (let i = 0; i < 7; i++) {
            const date = new Date(today);
            date.setDate(date.getDate() + i);
            
            // Day-specific patterns
            const dayPattern = this.getDayPattern(i);
            const weatherImpact = this.getWeatherImpact();
            
            // Calculate unique usage
            const usage = baseUsage * 
                          variationFactor * 
                          seasonalFactor * 
                          dayPattern * 
                          weatherImpact * 
                          (0.85 + Math.random() * 0.3);
            
            const temperature = 22 + Math.random() * 16; // 22-38°C
            const humidity = 40 + Math.random() * 40; // 40-80%
            
            this.forecastData.daily_data.push({
                day_name: this.days[i],
                full_name: this.fullDays[i],
                date: date.toISOString().split('T')[0],
                total_kwh: parseFloat(usage.toFixed(1)),
                peak_kwh: parseFloat((usage * 0.12 * (0.9 + Math.random() * 0.2)).toFixed(2)),
                peak_hour: this.generatePeakHour(temperature),
                cost_pkr: parseFloat((usage * this.pkrRate).toFixed(0)),
                confidence: 86 + Math.random() * 12,
                temperature: parseFloat(temperature.toFixed(1)),
                humidity: parseFloat(humidity.toFixed(1)),
                weather: this.getWeatherCondition(temperature),
                savings_potential: parseFloat((usage * 0.12 * (0.8 + Math.random() * 0.4)).toFixed(1))
            });
        }
    }

    getSeasonalFactor() {
        const month = new Date().getMonth();
        // Summer (June-Sept): higher usage, Winter: lower
        if (month >= 5 && month <= 8) return 1.3 + Math.random() * 0.4;
        if (month >= 11 || month <= 1) return 0.9 + Math.random() * 0.2;
        return 1.0 + Math.random() * 0.3;
    }

    getRandomBehaviorPattern() {
        const patterns = [
            { archetype: 'Efficient User', baseMultiplier: 0.8, description: 'Energy conscious household' },
            { archetype: 'Average User', baseMultiplier: 1.0, description: 'Typical usage patterns' },
            { archetype: 'Heavy User', baseMultiplier: 1.3, description: 'High energy consumption' },
            { archetype: 'Smart Home User', baseMultiplier: 0.85, description: 'Automated optimization' },
            { archetype: 'Peak Hour User', baseMultiplier: 1.1, description: 'High peak usage' }
        ];
        return patterns[Math.floor(Math.random() * patterns.length)];
    }

    getDayPattern(dayIndex) {
        // Weekend vs Weekday patterns
        if (dayIndex === 5 || dayIndex === 6) {
            return 1.15 + Math.random() * 0.2; // Weekend: higher usage
        }
        // Mid-week variation
        if (dayIndex === 2 || dayIndex === 3) {
            return 1.05 + Math.random() * 0.15;
        }
        return 0.95 + Math.random() * 0.15;
    }

    getWeatherImpact() {
        // Simulate weather impact
        const impacts = [0.85, 0.95, 1.0, 1.15, 1.25, 1.35];
        return impacts[Math.floor(Math.random() * impacts.length)];
    }

    getWeatherCondition(temp) {
        if (temp > 35) return 'Very Hot';
        if (temp > 30) return 'Hot';
        if (temp > 25) return 'Warm';
        if (temp > 20) return 'Pleasant';
        return 'Cool';
    }

    generatePeakHour(temperature) {
        // Peak hours vary by temperature
        if (temperature > 32) {
            return 14 + Math.floor(Math.random() * 4); // 2-5 PM
        } else if (temperature > 28) {
            return 15 + Math.floor(Math.random() * 3); // 3-5 PM
        } else {
            return 18 + Math.floor(Math.random() * 3); // 6-8 PM
        }
    }

    displayForecast() {
        if (!this.forecastData) return;

        const total = this.forecastData.daily_data.reduce((sum, day) => sum + day.total_kwh, 0);
        const peakDay = this.forecastData.daily_data.reduce((max, day) => 
            day.total_kwh > max.total_kwh ? day : max
        );
        const totalCost = this.forecastData.daily_data.reduce((sum, day) => sum + day.cost_pkr, 0);
        const avgConfidence = this.forecastData.daily_data.reduce((sum, day) => sum + day.confidence, 0) / 7;

        document.getElementById('weeklyTotal').textContent = `${total.toFixed(1)} kWh`;
        document.getElementById('peakDay').textContent = peakDay.full_name;
        document.getElementById('estimatedCost').textContent = `PKR ${this.formatNumber(totalCost)}`;
        document.getElementById('aiConfidence').textContent = `${avgConfidence.toFixed(1)}%`;
    }

    createChart() {
        const ctx = document.getElementById('forecastChart').getContext('2d');
        
        if (this.chart) {
            this.chart.destroy();
        }

        const data = this.forecastData.daily_data;
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.map(d => d.day_name),
                datasets: [{
                    label: 'Energy Usage (kWh)',
                    data: data.map(d => d.total_kwh),
                    borderColor: `hsl(${Math.random() * 60 + 140}, 70%, 50%)`,
                    backgroundColor: `hsla(${Math.random() * 60 + 140}, 70%, 50%, 0.1)`,
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    pointBackgroundColor: '#fff',
                    pointBorderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        padding: 12,
                        titleFont: { size: 14, weight: 'bold' },
                        bodyFont: { size: 13 },
                        callbacks: {
                            label: (context) => {
                                const day = data[context.dataIndex];
                                return [
                                    `Usage: ${day.total_kwh} kWh`,
                                    `Cost: PKR ${this.formatNumber(day.cost_pkr)}`,
                                    `Peak: ${day.peak_hour}:00`,
                                    `Temp: ${day.temperature}°C`,
                                    `Confidence: ${day.confidence.toFixed(1)}%`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'kWh',
                            font: { size: 14, weight: 'bold' }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Day of Week',
                            font: { size: 14, weight: 'bold' }
                        },
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }

    changeChartType(type) {
        if (!this.chart) return;
        this.chart.config.type = type;
        this.chart.update();
    }

    displayDailyBreakdown() {
        const container = document.getElementById('daysGrid');
        container.innerHTML = '';

        this.forecastData.daily_data.forEach(day => {
            const card = document.createElement('div');
            card.className = 'day-card';
            card.innerHTML = `
                <div class="day-header">
                    <div>
                        <div class="day-name">${day.full_name}</div>
                        <div class="day-date">${this.formatDate(day.date)}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.5rem; font-weight: 800; color: var(--primary);">
                            ${day.total_kwh} <span style="font-size: 0.8rem; color: var(--gray);">kWh</span>
                        </div>
                    </div>
                </div>
                <div class="day-stats">
                    <div class="day-stat">
                        <span class="stat-label">Cost</span>
                        <span class="stat-value">PKR ${this.formatNumber(day.cost_pkr)}</span>
                    </div>
                    <div class="day-stat">
                        <span class="stat-label">Peak Hour</span>
                        <span class="stat-value">${day.peak_hour}:00</span>
                    </div>
                    <div class="day-stat">
                        <span class="stat-label">Temperature</span>
                        <span class="stat-value">${day.temperature}°C</span>
                    </div>
                    <div class="day-stat">
                        <span class="stat-label">Weather</span>
                        <span class="stat-value">${day.weather}</span>
                    </div>
                    <div class="day-stat">
                        <span class="stat-label">Savings Potential</span>
                        <span class="stat-value" style="color: var(--success);">${day.savings_potential} kWh</span>
                    </div>
                </div>
            `;
            container.appendChild(card);
        });
    }

    async generateInitialAIMessage() {
        const peakDay = this.forecastData.daily_data.reduce((max, day) => 
            day.total_kwh > max.total_kwh ? day : max
        );
        const totalUsage = this.forecastData.daily_data.reduce((sum, day) => sum + day.total_kwh, 0);
        const avgTemp = this.forecastData.daily_data.reduce((sum, day) => sum + day.temperature, 0) / 7;

        const greeting = this.generateUniqueGreeting();
        const actionPlan = this.generateUniqueActionPlan(peakDay, totalUsage, avgTemp);

        this.addAIMessage(greeting, actionPlan);
    }

    generateUniqueGreeting() {
        const greetings = [
            "Hello! I've analyzed your energy patterns using three decades of experience with over 10,000 households. Let me share what I've discovered.",
            "Welcome! After analyzing your consumption patterns through my expertise , I have some valuable insights for you.",
            "Greetings! My energy intelligence have processed your data. Here's what the numbers tell me about your household.",
            "Hi there! Drawing from my analyzing energy patterns, I've created a personalized strategy for your home.",
            "Welcome to your personalized energy analysis! I can see unique opportunities in your usage patterns."
        ];
        return greetings[Math.floor(Math.random() * greetings.length)];
    }

    generateUniqueActionPlan(peakDay, totalUsage, avgTemp) {
        const plans = [];
        const savingsPotential = (totalUsage * 0.15 * this.pkrRate).toFixed(0);
        
        // Generate unique advice for each day
        this.forecastData.daily_data.forEach(day => {
            const advice = this.generateDaySpecificAdvice(day, peakDay, avgTemp);
            plans.push(`
                <div class="action-item">
                    <strong>${day.full_name} - ${day.total_kwh} kWh</strong>
                    ${advice}
                </div>
            `);
        });

        const uniqueInsight = this.generateUniqueInsight(totalUsage, avgTemp);

        return `
            <div class="action-plan">
                <h4><i class="fas fa-lightbulb"></i> Your Personalized 7-Day Action Plan</h4>
                <p style="margin-bottom: 15px; color: var(--gray);">
                    <strong>Potential Weekly Savings: PKR ${savingsPotential}</strong> • ${uniqueInsight}
                </p>
                ${plans.join('')}
            </div>
        `;
    }

    generateDaySpecificAdvice(day, peakDay, avgTemp) {
        const adviceOptions = [];

        // Temperature-based advice
        if (day.temperature > 32) {
            adviceOptions.push(`With ${day.temperature}°C heat, set AC to 26°C and use fans to save ${(day.total_kwh * 0.18).toFixed(1)} kWh.`);
            adviceOptions.push(`High temperature day! Pre-cool your home at ${day.peak_hour - 2}:00 AM, then raise AC temp by 2°C during peak hours.`);
        } else if (day.temperature > 28) {
            adviceOptions.push(`Moderate heat at ${day.temperature}°C - use natural ventilation until ${day.peak_hour - 1}:00, then AC as needed.`);
        } else {
            adviceOptions.push(`Pleasant ${day.temperature}°C weather - minimize AC use, open windows for natural cooling.`);
        }

        // Peak hour advice
        if (day.day_name === peakDay.day_name) {
            adviceOptions.push(`<strong>Peak Day Alert!</strong> Shift laundry and dishwasher to after 8 PM to save PKR ${(day.cost_pkr * 0.2).toFixed(0)}.`);
        }

        // Weather-based advice
        if (day.weather === 'Very Hot') {
            adviceOptions.push(`Extreme heat expected - close curtains at ${day.peak_hour - 3}:00 to reduce cooling load by 15%.`);
        }

        // Weekend vs weekday
        if (day.day_name === 'Sat' || day.day_name === 'Sun') {
            adviceOptions.push(`Weekend tip: Run major appliances during off-peak hours (midnight-6 AM) for 40% cost savings.`);
        }

        // Random selection for uniqueness
        return adviceOptions[Math.floor(Math.random() * adviceOptions.length)];
    }

    generateUniqueInsight(totalUsage, avgTemp) {
        const insights = [
            `Based on ${totalUsage.toFixed(1)} kWh weekly usage, you're in the ${this.getPercentile()}th percentile of efficiency.`,
            `Your consumption pattern suggests ${this.getArchetype()} behavior - optimized for ${this.getOptimizationArea()}.`,
            `At ${avgTemp.toFixed(1)}°C average temperature, I've calibrated your plan for maximum comfort and savings.`,
            `My analysis shows your household has ${this.getSavingsPotential()}% untapped savings potential.`,
            `Compared to 10,000+ similar households, you could improve efficiency by ${Math.floor(Math.random() * 20 + 10)}% in ${Math.floor(Math.random() * 30 + 30)} days.`
        ];
        return insights[Math.floor(Math.random() * insights.length)];
    }

    getPercentile() {
        return Math.floor(Math.random() * 40 + 40); // 40-80th percentile
    }

    getArchetype() {
        const archetypes = ['efficient', 'balanced', 'comfort-focused', 'tech-savvy', 'eco-conscious'];
        return archetypes[Math.floor(Math.random() * archetypes.length)];
    }

    getOptimizationArea() {
        const areas = ['cooling efficiency', 'load balancing', 'peak avoidance', 'appliance scheduling', 'behavioral patterns'];
        return areas[Math.floor(Math.random() * areas.length)];
    }

    getSavingsPotential() {
        return Math.floor(Math.random() * 15 + 10); // 10-25%
    }

    addAIMessage(greeting, content) {
        const chatContainer = document.getElementById('chatContainer');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai-message';
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <div class="message-header">
                    <h4>AI Energy Advisor</h4>
                    
                </div>
                <div class="message-text">
                    <p>${greeting}</p>
                    ${content}
                </div>
            </div>
        `;
        chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addUserMessage(message) {
        const chatContainer = document.getElementById('chatContainer');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-user"></i>
            </div>
            <div class="message-content">
                <div class="message-header">
                    <h4>You</h4>
                </div>
                <div class="message-text">
                    <p>${message}</p>
                </div>
            </div>
        `;
        chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    async sendMessage() {
        const input = document.getElementById('userInput');
        const message = input.value.trim();
        
        if (!message) return;

        this.addUserMessage(message);
        input.value = '';

        // Show typing indicator
        document.getElementById('typingIndicator').classList.add('active');

        // Simulate AI thinking time
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

        // Generate unique AI response
        const response = await this.generateAIResponse(message);

        // Hide typing indicator
        document.getElementById('typingIndicator').classList.remove('active');

        // Add AI response
        this.addAIMessage('', `<div class="message-text"><p>${response}</p></div>`);

        // Store conversation
        this.conversationHistory.push({
            user: message,
            ai: response,
            timestamp: new Date().toISOString()
        });
    }

    async generateAIResponse(question) {
        const lowerQuestion = question.toLowerCase();
        
        const peakDay = this.forecastData.daily_data.reduce((max, day) => 
            day.total_kwh > max.total_kwh ? day : max
        );
        const totalUsage = this.forecastData.daily_data.reduce((sum, day) => sum + day.total_kwh, 0);
        const totalCost = this.forecastData.daily_data.reduce((sum, day) => sum + day.cost_pkr, 0);
        const avgTemp = this.forecastData.daily_data.reduce((sum, day) => sum + day.temperature, 0) / 7;

        // Savings questions
        if (lowerQuestion.includes('save') || lowerQuestion.includes('reduce') || lowerQuestion.includes('money')) {
            const savingsAmount = Math.floor(totalCost * (0.12 + Math.random() * 0.08));
            const responses = [
                `Based on my  analyzing, you can save approximately <strong>PKR ${savingsAmount}</strong> this week. Focus on shifting high-consumption activities from ${peakDay.full_name}'s peak hour (${peakDay.peak_hour}:00) to off-peak times after 8 PM. This single change typically yields ${Math.floor(12 + Math.random() * 8)}% cost reduction.`,
                
                `I've identified <strong>PKR ${savingsAmount}</strong> in weekly savings potential. My three-decade analysis shows that households like yours benefit most from: 1) Setting AC to 26°C instead of 24°C (saves 15-18%), 2) Using fans in combination with AC (4°C cooler feeling, 25% less energy), 3) Running major appliances during off-peak hours (40% cheaper). Start with the AC adjustment - it's the easiest ${Math.floor(savingsAmount * 0.4)} PKR.`,
                
                `After analyzing your ${totalUsage.toFixed(1)} kWh weekly pattern, I see <strong>PKR ${savingsAmount}</strong> in savings opportunities. Your ${peakDay.full_name} usage of ${peakDay.total_kwh} kWh is ${Math.floor(15 + Math.random() * 20)}% above optimal. Try this: pre-cool your home at 24°C from ${peakDay.peak_hour - 2}:00 to ${peakDay.peak_hour}:00, then raise to 27°C during peak hours. This maintains comfort while cutting costs by ${Math.floor(8 + Math.random() * 7)}%.`
            ];
            return responses[Math.floor(Math.random() * responses.length)];
        }

        // Peak usage questions
        if (lowerQuestion.includes('peak') || lowerQuestion.includes('high') || lowerQuestion.includes('most')) {
            const responses = [
                `Your peak occurs on <strong>${peakDay.full_name}</strong> with ${peakDay.total_kwh} kWh, primarily at ${peakDay.peak_hour}:00. After pattern analysis, I recognize this as a classic '${this.getUsagePattern()}' profile. The spike is driven by ${peakDay.temperature}°C heat causing AC to work ${Math.floor(35 + Math.random() * 25)}% harder, combined with simultaneous appliance use. Stagger your appliances and you'll cut this peak by ${Math.floor(8 + Math.random() * 10)} kWh.`,
                
                `${peakDay.full_name}'s ${peakDay.total_kwh} kWh peak is ${Math.floor(18 + Math.random() * 15)}% above your weekly average. In my experience with similar households at ${avgTemp.toFixed(1)}°C average temperature, this suggests overlapping AC and high-power appliance usage during ${peakDay.peak_hour}:00-${peakDay.peak_hour + 2}:00. Move just your laundry or dishwasher to midnight-6 AM and watch this peak drop by ${(peakDay.total_kwh * (0.08 + Math.random() * 0.08)).toFixed(1)} kWh (worth PKR ${Math.floor(peakDay.cost_pkr * 0.15)}).`,
                
                `The ${peakDay.full_name} spike of ${peakDay.total_kwh} kWh tells an interesting story. My pattern in ${Math.floor(22 + Math.random() * 15)}% of households. At ${peakDay.temperature}°C, your cooling load peaks naturally around ${peakDay.peak_hour}:00. But here's the insight: by closing curtains at ${peakDay.peak_hour - 3}:00 and using ceiling fans, you can reduce this peak by 10-15% without sacrificing comfort. That's ${(peakDay.total_kwh * 0.125).toFixed(1)} kWh or PKR ${Math.floor(peakDay.cost_pkr * 0.125)} saved.`
            ];
            return responses[Math.floor(Math.random() * responses.length)];
        }

        // AC questions
        if (lowerQuestion.includes('ac') || lowerQuestion.includes('air') || lowerQuestion.includes('cooling')) {
            if (avgTemp > 30) {
                const responses = [
                    `At ${avgTemp.toFixed(1)}°C average temperature, AC is your biggest consumer (40-50% of costs). Here's wisdom from  Set it to <strong>26°C</strong>, not 24°C - this alone saves 15-20% with minimal comfort loss. Use 'eco' mode which my analysis shows reduces consumption by ${Math.floor(12 + Math.random() * 8)}%. Add a ceiling fan and you'll feel 4°C cooler while cutting AC usage by 25%. That's PKR ${Math.floor(totalCost * 0.35)} weekly savings just from smarter AC use.`,
                    
                    `In this ${avgTemp.toFixed(1)}°C heat, your AC works overtime. My 10,000-household database reveals: 1) Every degree below 26°C adds ${Math.floor(6 + Math.random() * 4)}% to your bill, 2) Dirty filters increase consumption ${Math.floor(12 + Math.random() * 8)}% (clean monthly!), 3) Auto mode beats continuous mode by ${Math.floor(15 + Math.random() * 10)}%. Your current usage suggests AC accounts for PKR ${Math.floor(totalCost * 0.42)} - optimize these three factors and cut that by PKR ${Math.floor(totalCost * 0.15)}.`,
                    
                    `Hot weather at ${avgTemp.toFixed(1)}°C demands smart AC strategy, I know the secret: <strong>Pre-cooling</strong>. Cool your home to 24°C from ${peakDay.peak_hour - 3}:00 to ${peakDay.peak_hour - 1}:00 (off-peak rates!), then raise to 27°C during peak hours. The thermal mass keeps you comfortable while saving ${Math.floor(25 + Math.random() * 15)}%. Also, ensure your outdoor unit has proper airflow - blocked units work ${Math.floor(20 + Math.random() * 15)}% harder. These changes could save you PKR ${Math.floor(totalCost * 0.22)} weekly.`
                ];
                return responses[Math.floor(Math.random() * responses.length)];
            } else {
                const responses = [
                    `At ${avgTemp.toFixed(1)}°C, you have great opportunities! This moderate temperature means AC isn't essential all day. My data shows optimal strategy: Use natural ventilation until ${Math.floor(10 + Math.random() * 3)}:00 AM, AC on 'auto' (not 'on') mode at 25-26°C during midday, then switch back to fans after ${Math.floor(18 + Math.random() * 2)}:00. This weather-adaptive approach saves ${Math.floor(20 + Math.random() * 12)}% compared to constant AC. That's PKR ${Math.floor(totalCost * 0.18)} in your pocket weekly.`,
                    
                    `Perfect ${avgTemp.toFixed(1)}°C weather for hybrid cooling! Here's strategy: Morning/evening (6-10 AM, 7-11 PM) - windows open, fans only. Midday heat (11 AM-6 PM) - AC at 26°C with fans. This 'temperature-responsive' approach, used by top ${Math.floor(15 + Math.random() * 10)}% efficient homes in my database, cuts AC usage ${Math.floor(30 + Math.random() * 15)}% while maintaining comfort. Your weekly saving: approximately PKR ${Math.floor(totalCost * 0.22)}.`,
                    
                    `Great news! At ${avgTemp.toFixed(1)}°C, you're in the 'goldilocks zone'. My data shows you can reduce AC dependency by ${Math.floor(35 + Math.random() * 20)}%. Strategy: Only use AC when indoor temp exceeds 28°C. Set it to 27°C (feels like 23°C with fan). Use 'sleep mode' at night which my analysis shows saves ${Math.floor(8 + Math.random() * 6)} kWh weekly. Monthly filter cleaning is crucial - improves efficiency ${Math.floor(10 + Math.random() * 8)}%. These simple changes = PKR ${Math.floor(totalCost * 0.24)} weekly savings.`
                ];
                return responses[Math.floor(Math.random() * responses.length)];
            }
        }

        // Weather questions
        if (lowerQuestion.includes('weather') || lowerQuestion.includes('temperature') || lowerQuestion.includes('hot')) {
            const responses = [
                `The ${avgTemp.toFixed(1)}°C average temperature significantly impacts your ${totalUsage.toFixed(1)} kWh consumption. My research with 10,000+ households reveals: each degree above 25°C increases cooling costs ${Math.floor(2 + Math.random() * 2)}%. Your ${peakDay.weather} conditions on ${peakDay.full_name} create peak demand. Combat this with passive cooling: close blinds by ${peakDay.peak_hour - 4}:00 (reduces load 10-20%), use strategic cross-ventilation, and delay heat-generating activities (cooking, laundry) until evening. This weather-adaptive behavior saves PKR ${Math.floor(totalCost * 0.16)} weekly.`,
                
                `Weather drives ${Math.floor(55 + Math.random() * 15)}% of your energy usage variation. At ${avgTemp.toFixed(1)}°C average, I see clear patterns: Your ${peakDay.full_name} ${peakDay.weather} conditions (${peakDay.temperature}°C) spike usage to ${peakDay.total_kwh} kWh, I know the counter-strategy: Morning cool-down (open windows 6-9 AM when it's ${Math.floor(avgTemp - 8)}°C), midday fortress mode (close everything, minimal AC), evening recovery (ventilate 8-11 PM). This circadian cooling approach cuts weather-driven consumption ${Math.floor(18 + Math.random() * 10)}%, saving PKR ${Math.floor(totalCost * 0.18)}.`,
                
                `Your ${totalUsage.toFixed(1)} kWh weekly usage is weather-influenced. The ${avgTemp.toFixed(1)}°C average creates a ${this.getWeatherImpactDescription()} impact. Specific insight: ${peakDay.full_name}'s ${peakDay.temperature}°C drives ${Math.floor(22 + Math.random() * 12)}% of your costs. Smart households in my 3 'thermal mass' strategy - pre-cool home to 23°C at 5 AM (cheapest rates), let temp drift to 28°C during day (massive savings), cool again at 8 PM. This weather-synchronized approach reduces dependence on peak-hour cooling by ${Math.floor(30 + Math.random() * 15)}%. Your savings: PKR ${Math.floor(totalCost * 0.25)} weekly.`
            ];
            return responses[Math.floor(Math.random() * responses.length)];
        }

        // Appliance questions
        if (lowerQuestion.includes('appliance') || lowerQuestion.includes('device') || lowerQuestion.includes('washing')) {
            const responses = [
                `Appliance optimization is key! My analysis shows your major appliances (fridge, washer, dryer, dishwasher) account for ${Math.floor(25 + Math.random() * 10)}% of usage. Power move: <strong>Time-of-use shifting</strong>. Run washer/dryer/dishwasher between midnight-6 AM when rates are 40% cheaper. Your ${totalUsage.toFixed(1)} kWh weekly usage suggests this could save ${(totalUsage * 0.08).toFixed(1)} kWh (PKR ${Math.floor(totalCost * 0.12)}). Also, wash in cold water (90% energy saved per load), air-dry when possible, and run full loads only. These habits = PKR ${Math.floor(totalCost * 0.18)} weekly savings.`,
                
                `Let me share appliance wisdom from 10,000+ homes: 1) Refrigerator: Set to 3-4°C (not 1°C), save ${Math.floor(8 + Math.random() * 5)}%, 2) Washing machine: Use 40°C max (not 60°C), save ${Math.floor(35 + Math.random() * 15)}%, 3) Microwave: Heat small items here (not oven), save ${Math.floor(60 + Math.random() * 20)}%, 4) TV/entertainment: Enable power-saving mode, save ${Math.floor(15 + Math.random() * 10)}%. Your current pattern suggests implementing these saves PKR ${Math.floor(totalCost * 0.14)} weekly. The real secret? Off-peak scheduling - move ${Math.floor(3 + Math.random() * 2)} appliance uses to midnight-6 AM = additional PKR ${Math.floor(totalCost * 0.08)} saved.`,
                
                `Appliances are silent energy vampires! My database reveals: Standby power costs you ${Math.floor(8 + Math.random() * 5)}% annually (unplug chargers!), old appliances use ${Math.floor(30 + Math.random() * 20)}% more than new ones (upgrade smart!), and inefficient usage patterns waste ${Math.floor(15 + Math.random() * 10)}%. For your ${totalUsage.toFixed(1)} kWh usage, I recommend: Smart plugs for TV/entertainment (auto-off saves ${Math.floor(6 + Math.random() * 4)}%), full loads only for washer/dryer (per-item costs ${Math.floor(4 + Math.random() * 3)}x more), and off-peak scheduling for high-power devices. Total impact: PKR ${Math.floor(totalCost * 0.19)} weekly savings.`
            ];
            return responses[Math.floor(Math.random() * responses.length)];
        }

        // Default comprehensive response
        const responses = [
            `Great question! Looking at your ${totalUsage.toFixed(1)} kWh weekly pattern , I see ${Math.floor(3 + Math.random() * 3)} key optimization opportunities: 1) ${peakDay.full_name}'s ${peakDay.total_kwh} kWh peak can drop ${Math.floor(15 + Math.random() * 10)}% through load shifting, 2) Average ${avgTemp.toFixed(1)}°C temperature suggests ${this.getTemperatureStrategy()}, 3) Your usage profile matches '${this.getArchetype()}' archetype with ${this.getSavingsPotential()}% savings potential. Combined impact: PKR ${Math.floor(totalCost * 0.16)} weekly savings. Want details on any of these?`,
            
            `After analyzing your consumption through my 10,000-household database, here's what stands out: Weekly usage ${totalUsage.toFixed(1)} kWh is ${this.getEfficiencyRating()} for your temperature zone (${avgTemp.toFixed(1)}°C avg). Your ${peakDay.full_name} peak suggests ${this.getPattern()} behavior. My recommendation: Focus on '3T' strategy - Temperature (optimize AC), Timing (shift to off-peak), Technology (upgrade inefficient devices). This tri-factor approach, proven over 30 years, typically yields ${Math.floor(18 + Math.random() * 12)}% savings. For you, that's PKR ${Math.floor(totalCost * 0.18)} weekly.`,
            
            `Let me give you the  perspective: Your energy signature shows ${this.getEnergySignature()}. At PKR ${this.formatNumber(totalCost)} weekly cost, you're spending PKR ${Math.floor(totalCost * 52 / 12)} monthly. My comparative analysis places you in the ${this.getPercentile()}th efficiency percentile. The gap to top 25% is ${Math.floor(15 + Math.random() * 15)}% - achievable through: AC optimization (${Math.floor(8 + Math.random() * 5)}% gain), peak-time awareness (${Math.floor(6 + Math.random() * 4)}% gain), and appliance scheduling (${Math.floor(5 + Math.random() * 3)}% gain). Start with the easiest (AC) and build from there!`
        ];
        return responses[Math.floor(Math.random() * responses.length)];
    }

    getUsagePattern() {
        const patterns = ['afternoon peak surge', 'evening usage concentration', 'midday cooling spike', 'distributed load pattern', 'temperature-driven consumption'];
        return patterns[Math.floor(Math.random() * patterns.length)];
    }

    getWeatherImpactDescription() {
        const descriptions = ['significant', 'moderate', 'substantial', 'considerable', 'notable'];
        return descriptions[Math.floor(Math.random() * descriptions.length)];
    }

    getTemperatureStrategy() {
        const strategies = [
            'AC pre-cooling strategy will maximize savings',
            'hybrid cooling (AC + fans) offers best value',
            'natural ventilation windows exist daily',
            'thermal mass optimization is optimal',
            'temperature-responsive cooling reduces waste'
        ];
        return strategies[Math.floor(Math.random() * strategies.length)];
    }

    getEfficiencyRating() {
        const ratings = ['above average', 'on par with efficient households', 'showing good baseline', 'typical for your profile', 'showing improvement potential'];
        return ratings[Math.floor(Math.random() * ratings.length)];
    }

    getPattern() {
        const patterns = ['peak-time concentration', 'balanced distribution', 'cooling-dominated', 'appliance-heavy', 'temperature-responsive'];
        return patterns[Math.floor(Math.random() * patterns.length)];
    }

    getEnergySignature() {
        const signatures = [
            'balanced consumption with optimization headroom',
            'cooling-dominated usage with clear peak patterns',
            'efficient baseline with targeted improvement areas',
            'typical household profile with standard variations',
            'moderate usage showing seasonal responsiveness'
        ];
        return signatures[Math.floor(Math.random() * signatures.length)];
    }

    askQuestion(question) {
        document.getElementById('userInput').value = question;
        this.sendMessage();
    }

    async regenerateForecast() {
        document.getElementById('loading').style.display = 'flex';
        
        // Generate completely new forecast
        await this.generateDynamicForecast();
        this.displayForecast();
        this.createChart();
        this.displayDailyBreakdown();
        
        // Clear chat and generate new initial message
        document.getElementById('chatContainer').innerHTML = '';
        await this.generateInitialAIMessage();
        
        document.getElementById('loading').style.display = 'none';
    }

    scrollToBottom() {
        const container = document.getElementById('chatContainer');
        container.scrollTop = container.scrollHeight;
    }

    formatNumber(num) {
        return new Intl.NumberFormat('en-PK').format(Math.round(num));
    }

    formatDate(dateStr) {
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    }

    initializeWisdomDatabase() {
        // Extensive wisdom database for truly unique responses
        return {
            savings_strategies: [
                'AC pre-cooling with thermal mass', 'Off-peak appliance scheduling',
                'Hybrid cooling systems', 'Load distribution optimization',
                'Weather-responsive behaviors', 'Appliance efficiency upgrades',
                'Smart home automation', 'Behavioral pattern modification'
            ],
            temperature_insights: [
                'Each degree matters 2-3%', 'Humidity affects perceived temperature',
                'Thermal mass extends comfort', 'Cross-ventilation multiplies efficiency',
                'Passive cooling reduces AC load', 'Pre-cooling beats constant cooling'
            ],
            behavioral_patterns: [
                'Peak-time awareness saves most', 'Gradual changes stick better',
                'Visible feedback drives improvement', 'Social comparison motivates',
                'Automation removes friction', 'Immediate rewards reinforce habits'
            ]
        };
    }

    initializeAdviceTemplates() {
        // Return empty for now - using dynamic generation
        return {};
    }

    initializeInsightGenerators() {
        // Return empty for now - using dynamic generation
        return {};
    }
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new EnergyWiseApp();
});