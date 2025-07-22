Universal Database Matrix

(https://github.com/your-username/Universal-Database-Matrix/actions/workflows/deploy.yml/badge.svg)](https://github.com/your-username/Universal-Database-Matrix/actions/workflows/deploy.yml)

The Universal Database Matrix is a complete, enterprise-grade FOSS database and graph analytics platform designed for maximum efficiency and accessibility. It transforms complex data operations into an intuitive, conversational experience through a unique Windows XP-inspired UI. 

The entire stack is packaged for a one-command deployment on Hetzner Cloud, delivering a production-ready environment in under 10 minutes for a fraction of the cost of major cloud providers.

✨ Key Features

Complete FOSS Stack: Integrates a full suite of powerful tools including PostgreSQL, Apache AGE for graph queries, PostGIS, MADlib for in-database machine learning, and NetworkX.  One-Command Deployment: A single shell script provisions the Hetzner Cloud infrastructure, installs K3s, and deploys the entire application stack.  

Extreme Cost-Efficiency: Run a full production-grade setup on Hetzner Cloud for approximately **€16.46/month**, an 88-94% cost reduction compared to equivalent setups on AWS, Azure, or GCP.  

Windows XP-Inspired UI: A familiar dual-layer interface that allows users to seamlessly toggle between a traditional Database Layer and a visual Graph Layer. 

Intelligent Configuration: A web-based configuration management interface with a wizard to simplify editing settings for KEDA, Prometheus, Grafana, and other tools.  

Automated Resource Scaling: Pre-configured KEDA auto-scaling responds to application load, scaling resources up and down to ensure performance and cost-efficiency.

Built-in Orchestration & Monitoring: Leverages Prefect for complex, cascading workflow automation and a Prometheus/Grafana stack for comprehensive, real-time monitoring.

## 🏗️ Architecture

The system runs a multi-pod architecture on a lightweight K3s cluster. This separates concerns, allowing components to be managed and scaled independently.

Deployment: Single-command deployment script using pre-configured Kubernetes manifests.  

UI Layer: A Node.js application serves the Windows XP-style interface with its dual-layer toggle system.

Data Layer: A stateful set runs a Hetzner-optimized instance of PostgreSQL with the Apache AGE extension enabled, using persistent volumes for storage.

Automation & Scaling: KEDA and Prefect run as separate deployments, interacting with the main application and database based on pre-defined workflows and scaling triggers.

🚀 Getting Started: 5-Minute Deployment

Prerequisites-

1.  A Hetzner Cloud account.
2.  Your Hetzner Cloud API Token.

Deployment-

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/Universal-Database-Matrix.git](https://github.com/your-username/Universal-Database-Matrix.git)
    cd Universal-Database-Matrix
    ```

2.  Set your Hetzner API Token:
    ```bash
    export HCLOUD_TOKEN="your-hetzner-cloud-token"
    ```

3.  Run the deployment script:
    ```bash
    ./deploy-hetzner.sh
    ```
    The script will create the server, install K3s, and deploy all application components. This process typically takes 5-10 minutes.

4.  Access Your System:
At the end of the script, the access URL will be displayed.

🔧 Configuration

Initial configuration is handled automatically by the deployment script. 

For ongoing management:

Easy Mode: Use the integrated Configuration Management web interface to edit configs with a wizard.

Advanced Mode: Modify the YAML and configuration files in the `config/` and `manifests/` directories and redeploy by pushing to your `main` branch (if CI/CD is configured).

🛠️ Technology Stack-

| Category                  | Technology                                                     |
| ------------------------- | -------------------------------------------------------------- |
| Cloud & Orchestration | Hetzner Cloud, K3s, KEDA                                       |
| Database | Neon PostgreSQL, Apache AGE, PostGIS                           |
| Analytics & ML | MADlib, NetworkX                                               |
| Workflow Automation | Prefect                                                        |
| Monitoring | Prometheus, Grafana                                            |
| Backend | Node.js (Express)                                              |
| Frontend | Custom Windows XP UI, D3.js/Cytoscape                          |

📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.


## 📄 License

This project is available under dual licensing:

- **GPL v3.0**: For open source projects and personal use (see `LICENSE-GPL`)
- **Commercial License**: For proprietary applications and commercial use (see `LICENSE-COMMERCIAL`)

For commercial licensing inquiries, please contact: brian95240@users.noreply.github.com
