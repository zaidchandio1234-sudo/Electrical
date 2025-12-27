// EnergyWise Pro - Authentication System (FINAL)

class AuthService {
    constructor() {
        this.currentUser = null;
        this.init();
    }

    init() {
        this.checkLoginStatus();
        this.setupEventListeners();
        this.protectPages();
    }

    setupEventListeners() {
        const loginForm = document.getElementById('loginForm');
        if (loginForm) {
            loginForm.addEventListener('submit', e => this.handleLogin(e));
        }

        const registerForm = document.getElementById('registerForm');
        if (registerForm) {
            registerForm.addEventListener('submit', e => this.handleRegister(e));
        }

        document.querySelectorAll('.logout-btn').forEach(btn => {
            btn.addEventListener('click', () => this.logout());
        });
    }

    handleRegister(e) {
        e.preventDefault();

        const name = document.getElementById('name').value;
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;

        let users = JSON.parse(localStorage.getItem('energywise_users')) || [];

        if (users.find(u => u.email === email)) {
            this.showMessage(
                'Account already exists. Please login.',
                'warning'
            );
            return;
        }

        users.push({ name, email, password });
        localStorage.setItem('energywise_users', JSON.stringify(users));

        this.showMessage(
            'Account created successfully. Redirecting to login...',
            'success'
        );

        setTimeout(() => {
            window.location.href = 'login.html';
        }, 1500);
    }

    handleLogin(e) {
        e.preventDefault();

        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;

        const users = JSON.parse(localStorage.getItem('energywise_users')) || [];
        const user = users.find(
            u => u.email === email && u.password === password
        );

        if (!user) {
            this.showMessage('Invalid email or password.', 'error');
            return;
        }

        this.setSession(user);
        window.location.href = 'dashboard.html';
    }

    setSession(user) {
        this.currentUser = user;
        localStorage.setItem('energywise_user', JSON.stringify(user));
        localStorage.setItem('energywise_login_time', Date.now());
    }

    checkLoginStatus() {
        const user = localStorage.getItem('energywise_user');
        const loginTime = localStorage.getItem('energywise_login_time');

        if (!user || !loginTime) return false;

        const age = Date.now() - loginTime;
        if (age > 24 * 60 * 60 * 1000) {
            this.logout();
            return false;
        }

        this.currentUser = JSON.parse(user);
        return true;
    }

    protectPages() {
        const protectedPages = ['dashboard.html'];

        if (
            protectedPages.some(p => location.pathname.includes(p)) &&
            !this.checkLoginStatus()
        ) {
            window.location.href = 'login.html';
        }
    }

    logout() {
        localStorage.removeItem('energywise_user');
        localStorage.removeItem('energywise_login_time');
        window.location.href = 'login.html';
    }

    showMessage(message, type) {
        const box = document.createElement('div');
        box.className = `msg ${type}`;
        box.textContent = message;
        document.body.appendChild(box);

        setTimeout(() => box.remove(), 4000);
    }
}

new AuthService();
