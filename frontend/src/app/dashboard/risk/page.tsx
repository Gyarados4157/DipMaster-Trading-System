'use client';

import { Suspense } from 'react';
import { RiskOverview } from '@/components/dashboard/risk/risk-overview';
import { RiskMetricsGrid } from '@/components/dashboard/risk/risk-metrics-grid';
import { RiskTimelineChart } from '@/components/dashboard/risk/risk-timeline-chart';
import { CorrelationMatrix } from '@/components/dashboard/risk/correlation-matrix';
import { RiskLimits } from '@/components/dashboard/risk/risk-limits';
import { VaRAnalysis } from '@/components/dashboard/risk/var-analysis';
import { PortfolioExposure } from '@/components/dashboard/risk/portfolio-exposure';
import { LoadingSkeleton } from '@/components/ui/loading-skeleton';

export default function RiskPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Risk Management</h1>
          <p className="text-muted-foreground">
            Monitor portfolio risk metrics and exposure limits
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="text-sm text-muted-foreground">
            Risk Score: <span className="font-medium text-trading-profit">Low</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-trading-profit rounded-full animate-pulse" />
            <span className="text-sm text-trading-profit font-medium">Monitoring</span>
          </div>
        </div>
      </div>

      {/* Risk Overview Cards */}
      <Suspense fallback={<LoadingSkeleton className="h-32" />}>
        <RiskOverview />
      </Suspense>

      {/* Main Risk Content */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          {/* Risk Metrics Grid */}
          <Suspense fallback={<LoadingSkeleton className="h-80" />}>
            <RiskMetricsGrid />
          </Suspense>

          {/* VaR Analysis */}
          <Suspense fallback={<LoadingSkeleton className="h-96" />}>
            <VaRAnalysis />
          </Suspense>
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          {/* Risk Timeline */}
          <Suspense fallback={<LoadingSkeleton className="h-80" />}>
            <RiskTimelineChart />
          </Suspense>

          {/* Portfolio Exposure */}
          <Suspense fallback={<LoadingSkeleton className="h-96" />}>
            <PortfolioExposure />
          </Suspense>
        </div>
      </div>

      {/* Bottom Section */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Correlation Matrix */}
        <Suspense fallback={<LoadingSkeleton className="h-96" />}>
          <CorrelationMatrix />
        </Suspense>

        {/* Risk Limits */}
        <Suspense fallback={<LoadingSkeleton className="h-96" />}>
          <RiskLimits />
        </Suspense>
      </div>
    </div>
  );
}