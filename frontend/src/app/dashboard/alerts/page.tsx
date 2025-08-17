'use client';

import { Suspense } from 'react';
import { AlertOverview } from '@/components/dashboard/alerts/alert-overview';
import { AlertHistory } from '@/components/dashboard/alerts/alert-history';
import { AlertFilters } from '@/components/dashboard/alerts/alert-filters';
import { AlertSettings } from '@/components/dashboard/alerts/alert-settings';
import { LoadingSkeleton } from '@/components/ui/loading-skeleton';

export default function AlertsPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Alert Management</h1>
          <p className="text-muted-foreground">
            Monitor system alerts and notifications
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="text-sm text-muted-foreground">
            Active alerts: <span className="font-medium text-trading-pending">3</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-trading-pending rounded-full animate-pulse" />
            <span className="text-sm text-trading-pending font-medium">Monitoring</span>
          </div>
        </div>
      </div>

      {/* Alert Overview */}
      <Suspense fallback={<LoadingSkeleton className="h-32" />}>
        <AlertOverview />
      </Suspense>

      {/* Main Content */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Left Column - Alert History */}
        <div className="xl:col-span-3 space-y-6">
          {/* Alert Filters */}
          <Suspense fallback={<LoadingSkeleton className="h-16" />}>
            <AlertFilters />
          </Suspense>

          {/* Alert History */}
          <Suspense fallback={<LoadingSkeleton className="h-96" />}>
            <AlertHistory />
          </Suspense>
        </div>

        {/* Right Column - Alert Settings */}
        <div className="space-y-6">
          <Suspense fallback={<LoadingSkeleton className="h-96" />}>
            <AlertSettings />
          </Suspense>
        </div>
      </div>
    </div>
  );
}