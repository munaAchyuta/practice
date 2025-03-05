# Integrating Goose with JIRA: Python Script Implementation Guide

Codename Goose, as an extensible AI agent framework, can connect with JIRA through its capability to execute Python scripts and interact with external APIs. This comprehensive guide provides the necessary steps and code to implement JIRA integration with Goose, allowing for automated issue tracking and project management tasks directly from your AI agent workflow.

## Understanding Goose's JIRA Integration Capabilities

Goose is designed with extensibility at its core, enabling integration with various external tools and services. According to the documentation, Goose can "interact with a multitude of tools via external APIs such as Jira, GitHub, Slack, infrastructure and data pipelines"[1]. This integration capability is facilitated through Goose's ability to execute shell commands and Python scripts, making it an ideal platform for automating JIRA operations that would traditionally require manual intervention.

The integration works by leveraging Python's JIRA libraries to communicate with your JIRA instance, allowing Goose to perform a wide range of operations programmatically. This approach combines the intelligence of Goose's AI agent with the robust project management capabilities of JIRA.

## Setting Up JIRA Python Integration

Before integrating with Goose, you'll need to establish the Python-JIRA connection. There are two primary methods for achieving this:

### Method 1: Using the JIRA Library

The JIRA library in Python provides a straightforward approach to interacting with your JIRA instance. Here's how to set it up:

### Authentication Setup

First, you need to generate an authentication token from your Atlassian account:

1. Log into your Atlassian account and navigate to the token generation page
2. Create and save your authentication token for use in your Python script
3. Install the JIRA library using pip:

```python
# Install the JIRA library
pip install jira
```

### Creating the Python Script for JIRA Integration

Below is a comprehensive Python script that establishes connection with JIRA and implements various common operations:

```python
from jira import JIRA
import pandas as pd

class JiraIntegration:
    def __init__(self, server_url, email, token):
        """
        Initialize JIRA connection using server URL and authentication credentials.
        
        Parameters:
            server_url (str): Your JIRA instance URL
            email (str): Your Atlassian account email
            token (str): Your generated API token
        """
        self.jira_options = {'server': server_url}
        self.jira = JIRA(options=self.jira_options, basic_auth=(email, token))
        print("JIRA connection established successfully.")
        
    def get_project_issues(self, project_key):
        """
        Retrieve all issues for a specific project.
        
        Parameters:
            project_key (str): The project key in JIRA (e.g., 'MED')
        
        Returns:
            DataFrame: Pandas DataFrame containing issue data
        """
        jql_str = f'project = {project_key}'
        issues = self.jira.search_issues(jql_str)
        
        issues_data = []
        for issue in issues:
            issue_dict = {
                'Key': issue.key,
                'Summary': issue.fields.summary,
                'Status': issue.fields.status.name,
                'Reporter': issue.fields.reporter.displayName,
                'Assignee': issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned",
                'Created': issue.fields.created,
                'Priority': issue.fields.priority.name
            }
            issues_data.append(issue_dict)
        
        return pd.DataFrame(issues_data)
    
    def create_issue(self, project_key, summary, description, issue_type='Task'):
        """
        Create a new JIRA issue.
        
        Parameters:
            project_key (str): The project key in JIRA
            summary (str): Issue summary/title
            description (str): Detailed issue description
            issue_type (str): Type of issue (Task, Bug, Story, etc.)
            
        Returns:
            Issue: The created JIRA issue object
        """
        issue_dict = {
            'project': {'key': project_key},
            'summary': summary,
            'description': description,
            'issuetype': {'name': issue_type},
        }
        
        new_issue = self.jira.create_issue(fields=issue_dict)
        print(f"Issue created successfully: {new_issue.key}")
        return new_issue
    
    def update_issue(self, issue_key, fields_dict):
        """
        Update an existing JIRA issue.
        
        Parameters:
            issue_key (str): The key of the issue to update
            fields_dict (dict): Dictionary of fields to update
            
        Returns:
            Issue: The updated JIRA issue object
        """
        issue = self.jira.issue(issue_key)
        issue.update(fields=fields_dict)
        print(f"Issue {issue_key} updated successfully.")
        return issue
    
    def assign_issue(self, issue_key, assignee_name):
        """
        Assign a JIRA issue to a user.
        
        Parameters:
            issue_key (str): The key of the issue to assign
            assignee_name (str): Username of the assignee
        """
        self.jira.assign_issue(issue_key, assignee_name)
        print(f"Issue {issue_key} assigned to {assignee_name} successfully.")
    
    def transition_issue(self, issue_key, transition_name):
        """
        Transition an issue to a different status.
        
        Parameters:
            issue_key (str): The key of the issue to transition
            transition_name (str): Name of the transition (e.g., 'In Progress')
        """
        issue = self.jira.issue(issue_key)
        transitions = self.jira.transitions(issue)
        
        transition_id = None
        for t in transitions:
            if t['name'].lower() == transition_name.lower():
                transition_id = t['id']
                break
        
        if transition_id:
            self.jira.transition_issue(issue, transition_id)
            print(f"Issue {issue_key} transitioned to '{transition_name}' successfully.")
        else:
            print(f"Transition '{transition_name}' not found for issue {issue_key}.")
    
    def add_comment(self, issue_key, comment_text):
        """
        Add a comment to a JIRA issue.
        
        Parameters:
            issue_key (str): The key of the issue
            comment_text (str): Text content of the comment
        """
        self.jira.add_comment(issue_key, comment_text)
        print(f"Comment added to issue {issue_key} successfully.")
    
    def get_issue_details(self, issue_key):
        """
        Get detailed information about a specific issue.
        
        Parameters:
            issue_key (str): The key of the issue
            
        Returns:
            dict: Dictionary containing issue details
        """
        issue = self.jira.issue(issue_key)
        
        details = {
            'Key': issue.key,
            'Summary': issue.fields.summary,
            'Description': issue.fields.description,
            'Status': issue.fields.status.name,
            'Reporter': issue.fields.reporter.displayName,
            'Assignee': issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned",
            'Created': issue.fields.created,
            'Updated': issue.fields.updated,
            'Priority': issue.fields.priority.name,
            'Issue Type': issue.fields.issuetype.name
        }
        
        return details

# Example usage
if __name__ == "__main__":
    # Replace with your actual JIRA instance details
    server_url = "https://your-domain.atlassian.net"
    email = "your-email@example.com"
    api_token = "your-api-token"
    
    # Initialize JIRA integration
    jira_integration = JiraIntegration(server_url, email, api_token)
    
    # Example: Get all issues from a project
    issues_df = jira_integration.get_project_issues("PROJECT")
    print(issues_df.head())
    
    # Example: Create a new issue
    new_issue = jira_integration.create_issue(
        "PROJECT", 
        "Test Issue via Python", 
        "This is a test issue created via Python script.", 
        "Task"
    )
    
    # Example: Assign the issue
    jira_integration.assign_issue(new_issue.key, "username")
    
    # Example: Add a comment
    jira_integration.add_comment(new_issue.key, "This is a test comment added via Python script.")
    
    # Example: Transition the issue to In Progress
    jira_integration.transition_issue(new_issue.key, "In Progress")
```

### Method 2: Using Personal Access Tokens (PAT)

For enhanced security, you can use Atlassian's Personal Access Tokens (PAT) instead of basic authentication. This method is particularly recommended for production environments[6]:

```python
from jira import JIRA

# Configure PAT authentication
jira = JIRA(
    options={'server': 'https://your-domain.atlassian.net'},
    token_auth='your-personal-access-token'
)

# Create a JIRA issue using PAT authentication
issue_dict = {
    'project': {'key': 'PROJECT'},
    'summary': 'Issue created with PAT authentication',
    'description': 'This issue was created using a Python script with PAT authentication.',
    'issuetype': {'name': 'Task'},
}

new_issue = jira.create_issue(fields=issue_dict)
print(f"Issue created successfully: {new_issue.key}")
```

## Integrating with Goose

Now that we have our JIRA integration script, we can utilize Goose's extensibility to incorporate this functionality into the AI agent workflow. Here's how to implement this integration:

### 1. Create a Goose Toolkit for JIRA

Goose uses toolkits to extend its functionality. Create a toolkit file for JIRA integration:

```python
# jira_toolkit.py
from goose import Toolkit
from jira import JIRA

class JiraToolkit(Toolkit):
    """Toolkit for interacting with JIRA from Goose."""
    
    def __init__(self, server_url, email, token):
        """Initialize the JIRA toolkit with authentication details."""
        self.jira_options = {'server': server_url}
        self.jira = JIRA(options=self.jira_options, basic_auth=(email, token))
    
    def get_project_issues(self, project_key):
        """Get all issues for a specific project."""
        jql_str = f'project = {project_key}'
        issues = self.jira.search_issues(jql_str)
        
        issues_data = []
        for issue in issues:
            issue_dict = {
                'Key': issue.key,
                'Summary': issue.fields.summary,
                'Status': issue.fields.status.name,
                'Reporter': issue.fields.reporter.displayName,
                'Assignee': issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned"
            }
            issues_data.append(issue_dict)
        
        return issues_data
    
    def create_issue(self, project_key, summary, description, issue_type='Task'):
        """Create a new JIRA issue."""
        issue_dict = {
            'project': {'key': project_key},
            'summary': summary,
            'description': description,
            'issuetype': {'name': issue_type},
        }
        
        new_issue = self.jira.create_issue(fields=issue_dict)
        return f"Issue created successfully: {new_issue.key}"
    
    def update_issue(self, issue_key, summary=None, description=None):
        """Update an existing JIRA issue."""
        fields_dict = {}
        if summary:
            fields_dict['summary'] = summary
        if description:
            fields_dict['description'] = description
        
        if fields_dict:
            issue = self.jira.issue(issue_key)
            issue.update(fields=fields_dict)
            return f"Issue {issue_key} updated successfully."
        return "No fields provided for update."
    
    def assign_issue(self, issue_key, assignee_name):
        """Assign a JIRA issue to a user."""
        self.jira.assign_issue(issue_key, assignee_name)
        return f"Issue {issue_key} assigned to {assignee_name} successfully."
    
    def add_comment(self, issue_key, comment_text):
        """Add a comment to a JIRA issue."""
        self.jira.add_comment(issue_key, comment_text)
        return f"Comment added to issue {issue_key} successfully."
```

### 2. Configure the Toolkit in Goose

After creating the toolkit, you need to configure Goose to use it. You can do this by:

1. Installing the toolkit in your Goose environment
2. Adding the necessary configuration for JIRA authentication

Here's a basic example of configuring Goose to use the JIRA toolkit from the command line:

```bash
# Install Python JIRA library for Goose to use
pip install jira

# Set up environment variables for JIRA authentication
export JIRA_SERVER_URL="https://your-domain.atlassian.net"
export JIRA_EMAIL="your-email@example.com"
export JIRA_TOKEN="your-api-token"

# Run Goose with JIRA toolkit
goose run --toolkit jira_toolkit.py
```

### 3. Using JIRA Functions Through Goose CLI

Once configured, you can instruct Goose to perform JIRA operations directly from the CLI:

```bash
# Get all issues from a project
goose "Get all issues from the PROJECT project and provide a summary"

# Create a new issue
goose "Create a new bug report in PROJECT with the title 'Authentication fails on mobile devices' and description 'Users are unable to log in from iOS and Android apps after the latest update.'"

# Update an issue status
goose "Move issue PROJECT-123 to 'In Progress' status"

# Add a comment to an issue
goose "Add a comment to PROJECT-123 saying 'The development team is investigating this issue.'"
```

## Advanced Usage with Goose

Beyond basic JIRA operations, Goose can perform more complex tasks by combining JIRA operations with other capabilities:

### Creating Issues Based on Code Analysis

Goose can analyze your codebase and automatically create JIRA issues for potential improvements:

```bash
goose "Analyze the src/authentication.js file, identify security vulnerabilities, and create JIRA issues for each vulnerability found in the PROJECT project"
```

### Automating Regular Reports

You can set up Goose to generate regular reports from JIRA data:

```bash
goose "Create a weekly summary report from JIRA PROJECT, showing all issues closed in the last 7 days, grouped by assignee, and email the report to the team"
```

### Integration with Development Workflows

Goose can help manage the entire development cycle by integrating JIRA with other tools:

```bash
goose "When I merge pull request #42 on GitHub, update the related JIRA issue PROJECT-123 to 'Ready for QA' status and assign it to the QA team lead"
```

## Conclusion

By leveraging Goose's ability to execute Python scripts and interact with external APIs, you can create a powerful integration with JIRA that automates project management tasks and streamlines your development workflow. The provided Python code and integration steps demonstrate how to establish this connection, enabling Goose to perform a wide range of JIRA operations programmatically.

This integration showcases the extensibility of the Codename Goose framework and its capability to enhance productivity through intelligent automation of routine tasks. As both Goose and your custom integration evolve, you can continue to expand the capabilities and use cases for this powerful combination of AI agent and project management tooling.

Citations:
[1] https://github.com/block/goose/blob/main/docs/index.md
[2] https://github.com/square/goose-vscode
[3] https://hevodata.com/learn/jira-python-integration/
[4] https://block.github.io/goose/blog/2025/01/28/introducing-codename-goose/
[5] https://langfuse.com/docs/integrations/goose
[6] https://confluence.atlassian.com/display/JIRAKB/How+to+create+Jira+issues+using+Python+script+with+PAT+(Personal+Access+Token)+as+the+authentication+mechanism
[7] https://www.youtube.com/watch?v=wUOPySw0VDw
[8] https://stackoverflow.com/questions/77045427/using-python-to-automate-creation-of-jira-tickets
[9] https://community.atlassian.com/t5/Jira-questions/Need-to-steps-to-connect-Jira-with-Python/qaq-p/1900696
[10] https://stackoverflow.com/questions/14078351/basic-authentication-with-jira-python
[11] https://community.atlassian.com/t5/Jira-questions/Accessing-Jira-with-python/qaq-p/1977811
[12] https://news.ycombinator.com/item?id=42879323
[13] https://jira.readthedocs.io
[14] https://block.github.io/goose/
[15] https://www.merge.dev/blog/how-to-get-all-issues-with-the-jira-api-in-python
[16] https://community.atlassian.com/t5/Jira-Software-questions/jira-extension/qaq-p/2115564
[17] https://github.com/pycontribs/jira
[18] https://slashdot.org/software/comparison/Jules-vs-codename-goose/
[19] https://www.tiktok.com/discover/chat-gpt-tasks
