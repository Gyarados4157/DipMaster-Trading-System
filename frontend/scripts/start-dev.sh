#!/bin/bash

# DipMaster Trading System - Frontend Development Startup Script
# This script sets up and starts the frontend development environment

set -e

echo "🚀 DipMaster Trading System - Frontend Startup"
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Node.js is installed
check_node() {
    print_status "Checking Node.js installation..."
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/"
        exit 1
    fi
    
    NODE_VERSION=$(node --version | cut -d'v' -f2)
    MAJOR_VERSION=$(echo $NODE_VERSION | cut -d'.' -f1)
    
    if [ "$MAJOR_VERSION" -lt 18 ]; then
        print_error "Node.js version 18+ is required. Current version: $NODE_VERSION"
        exit 1
    fi
    
    print_success "Node.js $NODE_VERSION detected"
}

# Check if npm is installed
check_npm() {
    print_status "Checking npm installation..."
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm"
        exit 1
    fi
    
    NPM_VERSION=$(npm --version)
    print_success "npm $NPM_VERSION detected"
}

# Setup environment file
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env.local" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env.local
            print_success "Created .env.local from .env.example"
            print_warning "Please review and update .env.local with your configuration"
        else
            print_warning ".env.example not found, creating basic .env.local"
            cat > .env.local << EOF
# DipMaster Frontend Configuration
BACKEND_URL=http://localhost:8000
WS_URL=ws://localhost:8000
JWT_SECRET=dipmaster-secret-key
NEXT_PUBLIC_ENABLE_DEMO_MODE=true
EOF
            print_success "Created basic .env.local"
        fi
    else
        print_success ".env.local already exists"
    fi
}

# Install dependencies
install_deps() {
    print_status "Installing dependencies..."
    
    if [ ! -d "node_modules" ]; then
        npm install
        print_success "Dependencies installed successfully"
    else
        print_status "Dependencies already installed, checking for updates..."
        npm update
        print_success "Dependencies updated"
    fi
}

# Run type checking
type_check() {
    print_status "Running TypeScript type checking..."
    if npm run type-check; then
        print_success "Type checking passed"
    else
        print_warning "Type checking found issues, but continuing..."
    fi
}

# Run linting
lint_check() {
    print_status "Running ESLint..."
    if npm run lint; then
        print_success "Linting passed"
    else
        print_warning "Linting found issues, but continuing..."
    fi
}

# Check backend connectivity
check_backend() {
    print_status "Checking backend connectivity..."
    
    BACKEND_URL=$(grep BACKEND_URL .env.local | cut -d'=' -f2)
    if [ -z "$BACKEND_URL" ]; then
        BACKEND_URL="http://localhost:8000"
    fi
    
    if curl -s -f "${BACKEND_URL}/health" > /dev/null 2>&1; then
        print_success "Backend is accessible at $BACKEND_URL"
    else
        print_warning "Backend is not accessible at $BACKEND_URL"
        print_warning "Make sure the DipMaster backend is running"
        print_warning "Frontend will start in demo mode"
    fi
}

# Start development server
start_dev() {
    print_status "Starting development server..."
    print_success "Frontend will be available at http://localhost:3000"
    print_status "Press Ctrl+C to stop the server"
    
    echo ""
    echo "🎯 Demo Credentials:"
    echo "   Username: admin"
    echo "   Password: dipmaster123"
    echo ""
    echo "📊 Features Available:"
    echo "   • Real-time dashboard"
    echo "   • Position monitoring"
    echo "   • Trading interface"
    echo "   • Risk analysis"
    echo "   • Market data"
    echo ""
    
    npm run dev
}

# Main execution
main() {
    echo ""
    print_status "Starting DipMaster frontend setup..."
    echo ""
    
    check_node
    check_npm
    setup_env
    install_deps
    type_check
    lint_check
    check_backend
    
    echo ""
    print_success "Setup completed successfully!"
    echo ""
    
    start_dev
}

# Handle script interruption
trap 'echo -e "\n${YELLOW}[INFO]${NC} Shutting down development server..."; exit 0' INT

# Run main function
main "$@"