'use client';

import { useState, useEffect } from 'react';
import { useTheme } from 'next-themes';
import { Button } from '@/components/ui/button';
import { 
  Sun, 
  Moon, 
  Monitor,
  Palette,
  Check
} from 'lucide-react';

interface ThemeToggleProps {
  className?: string;
  variant?: 'button' | 'dropdown' | 'compact';
}

export function ThemeToggle({ className, variant = 'button' }: ThemeToggleProps) {
  const { theme, setTheme, systemTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [showOptions, setShowOptions] = useState(false);

  // Avoid hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div className={`w-9 h-9 ${className}`} />; // Placeholder to prevent layout shift
  }

  const currentTheme = theme === 'system' ? systemTheme : theme;

  const themes = [
    {
      name: 'light',
      label: 'Light',
      icon: Sun,
      description: 'Light theme with bright backgrounds'
    },
    {
      name: 'dark',
      label: 'Dark',
      icon: Moon,
      description: 'Dark theme with reduced eye strain'
    },
    {
      name: 'system',
      label: 'System',
      icon: Monitor,
      description: 'Follow system preference'
    }
  ];

  // Compact variant for mobile/small spaces
  if (variant === 'compact') {
    return (
      <Button
        variant="ghost"
        size="sm"
        onClick={() => {
          const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
          setTheme(newTheme);
        }}
        className={`w-9 h-9 p-0 ${className}`}
        aria-label="Toggle theme"
      >
        {currentTheme === 'dark' ? (
          <Sun className="h-4 w-4" />
        ) : (
          <Moon className="h-4 w-4" />
        )}
      </Button>
    );
  }

  // Dropdown variant
  if (variant === 'dropdown') {
    return (
      <div className={`relative ${className}`}>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowOptions(!showOptions)}
          className="w-9 h-9 p-0"
          aria-label="Theme options"
        >
          <Palette className="h-4 w-4" />
        </Button>
        
        {showOptions && (
          <>
            {/* Backdrop */}
            <div 
              className="fixed inset-0 z-40" 
              onClick={() => setShowOptions(false)}
            />
            
            {/* Dropdown */}
            <div className="absolute right-0 top-10 z-50 w-48 rounded-lg border border-border bg-popover p-2 shadow-lg">
              <div className="space-y-1">
                {themes.map((themeOption) => {
                  const Icon = themeOption.icon;
                  const isSelected = theme === themeOption.name;
                  
                  return (
                    <button
                      key={themeOption.name}
                      onClick={() => {
                        setTheme(themeOption.name);
                        setShowOptions(false);
                      }}
                      className="flex w-full items-center justify-between rounded-md px-3 py-2 text-sm hover:bg-accent hover:text-accent-foreground transition-colors"
                    >
                      <div className="flex items-center space-x-2">
                        <Icon className="h-4 w-4" />
                        <div className="text-left">
                          <div className="font-medium">{themeOption.label}</div>
                          <div className="text-xs text-muted-foreground">
                            {themeOption.description}
                          </div>
                        </div>
                      </div>
                      
                      {isSelected && (
                        <Check className="h-4 w-4 text-primary" />
                      )}
                    </button>
                  );
                })}
              </div>
              
              {/* Theme Preview */}
              <div className="mt-3 pt-2 border-t border-border">
                <div className="text-xs text-muted-foreground mb-2">Preview:</div>
                <div className="grid grid-cols-3 gap-1">
                  <div className="h-8 rounded bg-background border border-border" />
                  <div className="h-8 rounded bg-card border border-border" />
                  <div className="h-8 rounded bg-muted border border-border" />
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    );
  }

  // Default button variant
  const currentThemeConfig = themes.find(t => t.name === (theme || 'system'));
  const Icon = currentThemeConfig?.icon || Monitor;

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => {
        const currentIndex = themes.findIndex(t => t.name === theme);
        const nextIndex = (currentIndex + 1) % themes.length;
        setTheme(themes[nextIndex].name);
      }}
      className={`relative group ${className}`}
      aria-label={`Current theme: ${currentThemeConfig?.label}. Click to cycle themes.`}
    >
      <Icon className="h-4 w-4 transition-transform group-hover:scale-110" />
      
      {/* Tooltip */}
      <div className="absolute bottom-full mb-2 left-1/2 transform -translate-x-1/2 px-2 py-1 bg-popover border border-border rounded text-xs opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50">
        {currentThemeConfig?.label} Theme
        <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-border" />
      </div>
    </Button>
  );
}

// Enhanced theme provider wrapper with additional features
export function EnhancedThemeProvider({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    // Apply system theme preference if supported
    if (typeof window !== 'undefined') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      
      // Listen for system theme changes
      const handleChange = (e: MediaQueryListEvent) => {
        // Only update if user has system theme selected
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'system' || !savedTheme) {
          document.documentElement.classList.toggle('dark', e.matches);
        }
      };

      mediaQuery.addEventListener('change', handleChange);
      
      return () => mediaQuery.removeEventListener('change', handleChange);
    }
  }, []);

  return <>{children}</>;
}

// Hook for theme-aware styling
export function useThemeColors() {
  const { theme, systemTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const isDark = mounted ? 
    (theme === 'dark' || (theme === 'system' && systemTheme === 'dark')) : 
    false;

  return {
    isDark,
    themeColors: {
      primary: isDark ? '#3b82f6' : '#1e40af',
      success: '#10b981',
      error: '#ef4444',
      warning: '#f59e0b',
      muted: isDark ? '#6b7280' : '#9ca3af',
    }
  };
}