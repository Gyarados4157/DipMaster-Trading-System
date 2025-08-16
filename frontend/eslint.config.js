const { createRequire } = require('module');
const require = createRequire(import.meta.url);

module.exports = {
  extends: [
    'next/core-web-vitals',
    '@typescript-eslint/recommended',
  ],
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint'],
  root: true,
  rules: {
    // TypeScript specific rules
    '@typescript-eslint/no-unused-vars': 'error',
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/prefer-const': 'error',
    
    // React specific rules
    'react-hooks/exhaustive-deps': 'warn',
    'react-hooks/rules-of-hooks': 'error',
    
    // General code quality
    'no-console': 'warn',
    'no-debugger': 'error',
    'prefer-const': 'error',
    'no-var': 'error',
    
    // Trading-specific linting
    'no-magic-numbers': ['warn', { 
      ignore: [0, 1, -1, 100, 1000],
      ignoreArrayIndexes: true,
      detectObjects: false
    }],
    
    // Performance
    'react/jsx-key': 'error',
    'react/no-array-index-key': 'warn',
  },
  overrides: [
    {
      files: ['*.ts', '*.tsx'],
      rules: {
        '@typescript-eslint/explicit-function-return-type': 'off',
        '@typescript-eslint/explicit-module-boundary-types': 'off',
      },
    },
    {
      files: ['**/*.test.ts', '**/*.test.tsx'],
      rules: {
        'no-magic-numbers': 'off',
        '@typescript-eslint/no-explicit-any': 'off',
      },
    },
  ],
  env: {
    browser: true,
    es2021: true,
    node: true,
  },
  settings: {
    react: {
      version: 'detect',
    },
  },
};