import { Component, type ErrorInfo, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
  title: string;
  message: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = {
    hasError: false,
  };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("UI PANEL CRASHED", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="state-card state-card-error">
          <p className="state-title">{this.props.title}</p>
          <p className="state-copy">{this.props.message}</p>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
