import { cn } from '@/lib/utils';

interface LoadingSkeletonProps {
  className?: string;
  lines?: number;
  showAvatar?: boolean;
}

export function LoadingSkeleton({ 
  className, 
  lines = 1, 
  showAvatar = false 
}: LoadingSkeletonProps) {
  return (
    <div className={cn('animate-pulse', className)}>
      <div className="space-y-3">
        {showAvatar && (
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 bg-muted rounded-full" />
            <div className="space-y-2">
              <div className="h-4 w-32 bg-muted rounded" />
              <div className="h-3 w-24 bg-muted rounded" />
            </div>
          </div>
        )}
        
        {Array.from({ length: lines }).map((_, i) => (
          <div key={i} className="space-y-2">
            <div className="h-4 bg-muted rounded w-full" />
            {i === 0 && lines > 1 && (
              <div className="h-4 bg-muted rounded w-5/6" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex space-x-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-5 bg-muted rounded flex-1" />
        ))}
      </div>
      
      {/* Rows */}
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex space-x-4">
          {Array.from({ length: 5 }).map((_, j) => (
            <div key={j} className="h-8 bg-muted rounded flex-1" />
          ))}
        </div>
      ))}
    </div>
  );
}

export function ChartSkeleton() {
  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div className="h-6 w-32 bg-muted rounded" />
        <div className="h-8 w-24 bg-muted rounded" />
      </div>
      
      <div className="h-64 bg-muted rounded-lg flex items-end justify-between px-4 pb-4">
        {Array.from({ length: 12 }).map((_, i) => (
          <div 
            key={i} 
            className="bg-muted-foreground/20 rounded-sm w-4"
            style={{ 
              height: `${Math.random() * 80 + 20}%` 
            }}
          />
        ))}
      </div>
    </div>
  );
}

export function CardSkeleton() {
  return (
    <div className="rounded-lg border bg-card p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div className="h-5 w-24 bg-muted rounded" />
        <div className="h-5 w-5 bg-muted rounded" />
      </div>
      
      <div className="space-y-2">
        <div className="h-8 w-20 bg-muted rounded" />
        <div className="h-4 w-32 bg-muted rounded" />
      </div>
      
      <div className="h-16 bg-muted rounded" />
    </div>
  );
}