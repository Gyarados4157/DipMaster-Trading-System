'use client';

import { Suspense } from 'react';
import { StrategyOverview } from '@/components/dashboard/strategy/strategy-overview';
import { ParameterConfiguration } from '@/components/dashboard/strategy/parameter-configuration';
import { PerformanceMetrics } from '@/components/dashboard/strategy/performance-metrics';
import { BacktestResults } from '@/components/dashboard/strategy/backtest-results';
import { SignalAnalysis } from '@/components/dashboard/strategy/signal-analysis';
import { TimingAnalysis } from '@/components/dashboard/strategy/timing-analysis';
import { LoadingSkeleton } from '@/components/ui/loading-skeleton';

export default function StrategyPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Strategy Analysis</h1>
          <p className="text-muted-foreground">
            DipMaster AI strategy performance and configuration
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="text-sm text-muted-foreground">
            Strategy Status: <span className="font-medium text-trading-profit">Active</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-trading-profit rounded-full animate-pulse" />
            <span className="text-sm text-trading-profit font-medium">Running</span>
          </div>
        </div>
      </div>

      {/* Strategy Overview */}
      <Suspense fallback={<LoadingSkeleton className="h-48" />}>
        <StrategyOverview />
      </Suspense>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left Column - Performance */}
        <div className="xl:col-span-2 space-y-6">
          {/* Performance Metrics */}
          <Suspense fallback={<LoadingSkeleton className="h-96" />}>
            <PerformanceMetrics />
          </Suspense>

          {/* Backtest Results */}
          <Suspense fallback={<LoadingSkeleton className="h-96" />}>
            <BacktestResults />
          </Suspense>

          {/* Signal Analysis */}
          <Suspense fallback={<LoadingSkeleton className="h-80" />}>
            <SignalAnalysis />
          </Suspense>
        </div>

        {/* Right Column - Configuration */}
        <div className="space-y-6">
          {/* Parameter Configuration */}
          <Suspense fallback={<LoadingSkeleton className="h-96" />}>
            <ParameterConfiguration />
          </Suspense>

          {/* Timing Analysis */}
          <Suspense fallback={<LoadingSkeleton className="h-80" />}>
            <TimingAnalysis />
          </Suspense>
        </div>
      </div>
    </div>
  );
}