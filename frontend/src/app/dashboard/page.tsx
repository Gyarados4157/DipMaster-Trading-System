'use client';

import { Suspense } from 'react';
import { OverviewStats } from '@/components/dashboard/overview-stats';
import { PnLChart } from '@/components/dashboard/pnl-chart';
import { ActivePositions } from '@/components/dashboard/active-positions';
import { RecentTrades } from '@/components/dashboard/recent-trades';
import { RiskMetrics } from '@/components/dashboard/risk-metrics';
import { MarketOverview } from '@/components/dashboard/market-overview';
import { SystemStatus } from '@/components/dashboard/system-status';
import { LoadingSkeleton } from '@/components/ui/loading-skeleton';

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Trading Dashboard</h1>
          <p className="text-muted-foreground">
            Monitor your DipMaster strategy performance in real-time
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="text-sm text-muted-foreground">
            Last updated: <span className="font-medium text-foreground">
              {new Date().toLocaleTimeString()}
            </span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-trading-profit rounded-full animate-pulse" />
            <span className="text-sm text-trading-profit font-medium">Live</span>
          </div>
        </div>
      </div>

      {/* Overview Stats */}
      <Suspense fallback={<LoadingSkeleton className="h-32" />}>
        <OverviewStats />
      </Suspense>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Charts */}
        <div className="lg:col-span-2 space-y-6">
          {/* PnL Chart */}
          <Suspense fallback={<LoadingSkeleton className="h-96" />}>
            <PnLChart />
          </Suspense>

          {/* Recent Trades */}
          <Suspense fallback={<LoadingSkeleton className="h-80" />}>
            <RecentTrades />
          </Suspense>
        </div>

        {/* Right Column - Side panels */}
        <div className="space-y-6">
          {/* System Status */}
          <Suspense fallback={<LoadingSkeleton className="h-40" />}>
            <SystemStatus />
          </Suspense>

          {/* Active Positions */}
          <Suspense fallback={<LoadingSkeleton className="h-60" />}>
            <ActivePositions />
          </Suspense>

          {/* Risk Metrics */}
          <Suspense fallback={<LoadingSkeleton className="h-60" />}>
            <RiskMetrics />
          </Suspense>
        </div>
      </div>

      {/* Bottom Section */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Market Overview */}
        <Suspense fallback={<LoadingSkeleton className="h-80" />}>
          <MarketOverview />
        </Suspense>

        {/* Strategy Performance */}
        <Suspense fallback={<LoadingSkeleton className="h-80" />}>
          <div className="chart-container">
            <h3 className="text-lg font-semibold mb-4">Strategy Performance</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-trading-profit">82.1%</div>
                <div className="text-sm text-muted-foreground">Win Rate</div>
              </div>
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-dipmaster-blue">87.9%</div>
                <div className="text-sm text-muted-foreground">Dip Buying Rate</div>
              </div>
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-dipmaster-orange">96min</div>
                <div className="text-sm text-muted-foreground">Avg Hold Time</div>
              </div>
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <div className="text-2xl font-bold text-dipmaster-green">100%</div>
                <div className="text-sm text-muted-foreground">Boundary Compliance</div>
              </div>
            </div>
          </div>
        </Suspense>
      </div>
    </div>
  );
}