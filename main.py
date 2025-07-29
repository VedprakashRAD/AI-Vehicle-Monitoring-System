"""
Main Application Entry Point
"""

from src.web.dashboard import VehicleDashboard


def main():
    """Start the Vehicle Monitoring Web Dashboard"""
    dashboard = VehicleDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()

