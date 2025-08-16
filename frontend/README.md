# DipMaster Trading System - Frontend

Professional cryptocurrency trading dashboard built with Next.js, providing real-time monitoring and control for the DipMaster AI trading strategy.

## ✨ Features

### 🎯 Core Features
- **Real-time Dashboard**: Live PnL, positions, and market data
- **Professional Charts**: Interactive trading charts with technical indicators
- **Risk Monitoring**: VaR, drawdown, and exposure analysis
- **Order Management**: Real-time order tracking and execution
- **Alert System**: Instant notifications for critical events
- **System Health**: Comprehensive monitoring and diagnostics

### 📊 Strategy Monitoring
- **82.1% Win Rate Tracking**: Real-time strategy performance
- **15-minute Boundary Management**: Precise exit timing
- **87.9% Dip Buying Rate**: Entry signal validation
- **Risk Metrics**: Comprehensive risk analysis and alerts

### 🎨 User Experience
- **Dark/Light Theme**: Professional trading interface
- **Responsive Design**: Mobile, tablet, and desktop optimized
- **Real-time Updates**: WebSocket-powered live data
- **Intuitive Navigation**: Clean and organized layout

## 🚀 Quick Start

### Prerequisites
- Node.js 18.0.0 or higher
- npm or yarn package manager
- DipMaster Backend API running

### Installation

1. **Clone and navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Environment setup**
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your configuration
   ```

4. **Run development server**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

5. **Open your browser**
   Navigate to [http://localhost:3000](http://localhost:3000)

### Demo Access
- **Username**: `admin`
- **Password**: `dipmaster123`

## 🏗️ Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js 14 App Router
│   │   ├── auth/              # Authentication pages
│   │   ├── dashboard/         # Main dashboard pages
│   │   ├── globals.css        # Global styles
│   │   ├── layout.tsx         # Root layout
│   │   └── page.tsx           # Home page
│   ├── components/            # React components
│   │   ├── dashboard/         # Dashboard-specific components
│   │   ├── ui/                # Reusable UI components
│   │   ├── providers.tsx      # App providers
│   │   └── theme-provider.tsx # Theme management
│   ├── hooks/                 # Custom React hooks
│   │   ├── use-api.ts         # API data fetching
│   │   ├── use-auth.ts        # Authentication
│   │   └── use-websocket.ts   # WebSocket connection
│   ├── lib/                   # Utility libraries
│   │   └── utils.ts           # Common utilities
│   └── types/                 # TypeScript type definitions
│       └── index.ts           # Type exports
├── public/                    # Static assets
├── tailwind.config.js         # Tailwind CSS configuration
├── tsconfig.json             # TypeScript configuration
└── next.config.js            # Next.js configuration
```

## 🔧 Configuration

### Environment Variables

Key environment variables in `.env.local`:

```bash
# Backend API
BACKEND_URL=http://localhost:8000
WS_URL=ws://localhost:8000

# Authentication
JWT_SECRET=your-secret-key

# Feature Flags
NEXT_PUBLIC_ENABLE_DEMO_MODE=true
NEXT_PUBLIC_ENABLE_SOUND_NOTIFICATIONS=true

# Trading Configuration
NEXT_PUBLIC_DEFAULT_SYMBOLS=BTCUSDT,ETHUSDT,ADAUSDT
NEXT_PUBLIC_MAX_POSITIONS=3
```

### API Endpoints

The frontend connects to these backend endpoints:

- `GET /api/pnl` - PnL time series data
- `GET /api/positions/latest` - Current positions
- `GET /api/trades` - Trading history
- `GET /api/risk/metrics` - Risk indicators
- `GET /api/system/health` - System status
- `WS /ws` - Real-time WebSocket updates

## 📱 Pages and Components

### Main Pages

1. **Dashboard** (`/dashboard`)
   - Overview statistics
   - PnL charts
   - Active positions
   - Recent trades

2. **Trading** (`/dashboard/trading`)
   - Live market data
   - Order management
   - Position controls

3. **Risk Monitor** (`/dashboard/risk`)
   - VaR analysis
   - Exposure breakdown
   - Risk alerts

4. **Analytics** (`/dashboard/analytics`)
   - Strategy performance
   - Execution analysis
   - Historical data

### Key Components

- **PnLChart**: Interactive profit/loss visualization
- **ActivePositions**: Real-time position monitoring
- **RiskMetrics**: Comprehensive risk dashboard
- **MarketOverview**: Multi-symbol market data
- **SystemStatus**: Health monitoring

## 🎨 Styling and Theming

### Tailwind CSS
- Custom color palette for trading applications
- Dark/light theme support
- Responsive design utilities
- Component-specific styling

### Theme Colors
```css
/* Trading Colors */
--trading-profit: #10b981    /* Green for profits */
--trading-loss: #ef4444      /* Red for losses */
--trading-pending: #f59e0b    /* Yellow for pending */
--trading-neutral: #6b7280    /* Gray for neutral */

/* DipMaster Brand Colors */
--dipmaster-blue: #1e40af
--dipmaster-green: #10b981
--dipmaster-orange: #f59e0b
--dipmaster-purple: #8b5cf6
```

## 🔄 Real-time Features

### WebSocket Integration
- **Price Updates**: Live market data streams
- **Position Changes**: Real-time position updates
- **Order Events**: Instant order status changes
- **System Alerts**: Critical notifications
- **Auto-reconnection**: Reliable connection management

### Data Refresh
- **PnL Data**: 30-second intervals
- **Positions**: 10-second intervals
- **Market Data**: 5-second intervals
- **System Health**: 30-second intervals

## 📊 Charts and Visualization

### Chart Types
- **Line Charts**: PnL trends and price movements
- **Area Charts**: Cumulative performance
- **Bar Charts**: Volume and histogram data
- **Progress Bars**: Risk metrics and usage

### Chart Features
- **Time Range Selection**: 1D, 1W, 1M, 3M, YTD, All
- **Interactive Tooltips**: Detailed data on hover
- **Responsive Design**: Adapts to screen size
- **Real-time Updates**: Live data streaming

## 🔐 Security Features

### Authentication
- **JWT Token Management**: Secure token handling
- **Route Protection**: Authenticated route guards
- **Session Management**: Automatic session handling
- **Logout Functionality**: Secure session termination

### Data Security
- **API Token Headers**: Secure API communication
- **Environment Variables**: Sensitive data protection
- **Input Validation**: Form and data validation
- **Error Handling**: Secure error messages

## 🚀 Performance Optimization

### Next.js Features
- **App Router**: Latest Next.js routing system
- **Server Components**: Optimized server rendering
- **Image Optimization**: Automatic image optimization
- **Code Splitting**: Automatic bundle optimization

### React Optimization
- **React Query**: Intelligent data caching
- **Lazy Loading**: Component lazy loading
- **Memoization**: Expensive operation caching
- **Virtual Scrolling**: Large list optimization

## 🧪 Development

### Scripts
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
npm run type-check   # TypeScript type checking
```

### Testing
```bash
npm run test         # Run tests
npm run test:watch   # Watch mode testing
```

### Code Quality
- **TypeScript**: Strict type checking
- **ESLint**: Code linting and formatting
- **Prettier**: Code formatting
- **Husky**: Git hooks for quality

## 📦 Deployment

### Production Build
```bash
npm run build
npm run start
```

### Docker Deployment
```bash
# Build image
docker build -t dipmaster-frontend .

# Run container
docker run -p 3000:3000 dipmaster-frontend
```

### Environment Setup
1. Copy `.env.example` to `.env.local`
2. Configure backend URLs
3. Set authentication secrets
4. Enable/disable features as needed

## 🔧 Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check backend API is running
   - Verify BACKEND_URL in environment
   - Check network connectivity

2. **Authentication Failed**
   - Verify JWT_SECRET matches backend
   - Check token expiration
   - Clear browser cache/cookies

3. **WebSocket Issues**
   - Check WS_URL configuration
   - Verify firewall settings
   - Check browser WebSocket support

4. **Performance Issues**
   - Reduce data refresh intervals
   - Enable production mode
   - Check network latency

## 📚 Documentation

### API Integration
- All API calls are handled through custom hooks
- Automatic error handling and retry logic
- Type-safe API responses
- Real-time data synchronization

### Component Architecture
- Modular component design
- Reusable UI components
- Custom hooks for data fetching
- Context providers for global state

### State Management
- Zustand for authentication state
- React Query for server state
- Local state for UI interactions
- WebSocket state management

## 🤝 Contributing

1. Follow TypeScript strict mode
2. Use provided ESLint configuration
3. Maintain responsive design principles
4. Test WebSocket connectivity
5. Document new components

## 📄 License

This project is part of the DipMaster Trading System. See the main project LICENSE file for details.

---

**⚠️ Important Notes:**
- This is a professional trading interface
- Always test in demo mode first
- Monitor API rate limits
- Keep authentication tokens secure
- Regularly update dependencies

**📞 Support:**
- Check system logs for errors
- Verify backend connectivity
- Review WebSocket connection status
- Monitor browser console for issues

---

**DipMaster Trading System v1.0.0**  
Professional Cryptocurrency Trading Dashboard  
Built with Next.js 14, TypeScript, and Tailwind CSS