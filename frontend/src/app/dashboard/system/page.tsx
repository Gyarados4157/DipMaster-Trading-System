'use client';

import { Suspense } from 'react';
import { SystemOverview } from '@/components/dashboard/system/system-overview';
import { PerformanceMetrics } from '@/components/dashboard/system/performance-metrics';
import { SystemLogs } from '@/components/dashboard/system/system-logs';
import { ResourceUsage } from '@/components/dashboard/system/resource-usage';
import { ServiceStatus } from '@/components/dashboard/system/service-status';
import { NetworkMonitor } from '@/components/dashboard/system/network-monitor';
import { LoadingSkeleton } from '@/components/ui/loading-skeleton';

export default function SystemPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">System Health</h1>
          <p className="text-muted-foreground">
            Monitor system performance and infrastructure status
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="text-sm text-muted-foreground">
            System Status: <span className="font-medium text-trading-profit">Healthy</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-trading-profit rounded-full animate-pulse" />
            <span className="text-sm text-trading-profit font-medium">Operational</span>
          </div>
        </div>
      </div>

      {/* System Overview */}
      <Suspense fallback={<LoadingSkeleton className="h-48" />}>
        <SystemOverview />
      </Suspense>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left Column - Performance */}
        <div className="xl:col-span-2 space-y-6">
          {/* Performance Metrics */}
          <Suspense fallback={<LoadingSkeleton className="h-96" />}>
            <PerformanceMetrics />
          </Suspense>

          {/* Resource Usage */}
          <Suspense fallback={<LoadingSkeleton className="h-80" />}>
            <ResourceUsage />
          </Suspense>

          {/* Network Monitor */}
          <Suspense fallback={<LoadingSkeleton className="h-80" />}>
            <NetworkMonitor />
          </Suspense>
        </div>

        {/* Right Column - Status and Logs */}
        <div className="space-y-6">
          {/* Service Status */}
          <Suspense fallback={<LoadingSkeleton className="h-96" />}>
            <ServiceStatus />
          </Suspense>

          {/* System Logs */}
          <Suspense fallback={<LoadingSkeleton className="h-96" />}>
            <SystemLogs />
          </Suspense>
        </div>
      </div>
    </div>
  );
}