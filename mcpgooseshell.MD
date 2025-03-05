# Integrating Goose with JIRA Using Shell Commands

Integrating Goose AI agent framework with JIRA via shell commands allows you to automate JIRA operations directly from your terminal. While the search results don't provide explicit integration steps specifically for JIRA, we can leverage Goose's extension capabilities and shell command execution to create a functional JIRA integration.

## Understanding Goose's Extension Capabilities

Goose supports external extensions through its CLI, allowing you to connect with applications and tools already in your workflow[3]. The framework provides multiple ways to add extensions:

1. Built-in extensions (Developer, Computer Controller, Memory, JetBrains, Google Drive)
2. Command-line extensions (local commands or scripts)
3. Remote extensions (connecting to remote systems via SSE)

For JIRA integration, we'll focus on the command-line extension approach, which enables us to run shell scripts that interact with JIRA's REST API.

## Setting Up JIRA Shell Integration

### Prerequisites

Before setting up the integration, ensure you have:

1. Goose installed on your system
2. JIRA access with appropriate permissions
3. cURL or similar HTTP client for API requests
4. Basic authentication credentials or API token for JIRA

### Creating JIRA Shell Scripts

Create a series of shell scripts that perform common JIRA operations using cURL. Here are examples of scripts for various JIRA operations:

#### 1. Create a Script for Authentication (`jira-auth.sh`)

```bash
#!/bin/bash

# Configuration variables
JIRA_URL="https://your-domain.atlassian.net"
JIRA_USER="your-email@example.com"
JIRA_API_TOKEN="your-api-token"

# Export variables for other scripts to use
export JIRA_URL
export JIRA_USER
export JIRA_API_TOKEN

# Test authentication
echo "Testing JIRA authentication..."
curl -s -D- -X GET -H "Content-Type: application/json" \
  -u "${JIRA_USER}:${JIRA_API_TOKEN}" \
  "${JIRA_URL}/rest/api/2/myself" | grep "HTTP/"

if [ $? -eq 0 ]; then
  echo "Authentication successful!"
else
  echo "Authentication failed. Please check your credentials."
  exit 1
fi
```

#### 2. Script to Get Issues from a Project (`jira-get-issues.sh`)

```bash
#!/bin/bash

# Source authentication variables
source ./jira-auth.sh

# Get project key from command line argument
PROJECT_KEY=$1

if [ -z "$PROJECT_KEY" ]; then
  echo "Usage: $0 "
  exit 1
fi

# Fetch issues from project
curl -s -X GET \
  -H "Content-Type: application/json" \
  -u "${JIRA_USER}:${JIRA_API_TOKEN}" \
  "${JIRA_URL}/rest/api/2/search?jql=project=${PROJECT_KEY}&maxResults=10" | \
  jq '.issues[] | {key: .key, summary: .fields.summary, status: .fields.status.name}'
```

#### 3. Script to Create a JIRA Issue (`jira-create-issue.sh`)

```bash
#!/bin/bash

# Source authentication variables
source ./jira-auth.sh

# Get parameters from command line
PROJECT_KEY=$1
SUMMARY=$2
DESCRIPTION=$3
ISSUE_TYPE=${4:-"Task"}

if [ -z "$PROJECT_KEY" ] || [ -z "$SUMMARY" ]; then
  echo "Usage: $0   [DESCRIPTION] [ISSUE_TYPE]"
  exit 1
fi

# Create JSON payload
JSON_DATA=$(cat   "
  exit 1
fi

# Create JSON based on field type
if [ "$FIELD_TYPE" == "status" ]; then
  # Get the transition ID for the status
  TRANSITIONS=$(curl -s -X GET \
    -H "Content-Type: application/json" \
    -u "${JIRA_USER}:${JIRA_API_TOKEN}" \
    "${JIRA_URL}/rest/api/2/issue/${ISSUE_KEY}/transitions")
  
  TRANSITION_ID=$(echo $TRANSITIONS | jq -r ".transitions[] | select(.name == \"$FIELD_VALUE\") | .id")
  
  if [ -z "$TRANSITION_ID" ]; then
    echo "Status '$FIELD_VALUE' not found for issue $ISSUE_KEY"
    exit 1
  fi
  
  JSON_DATA="{\"transition\": {\"id\": \"$TRANSITION_ID\"}}"
  ENDPOINT="${JIRA_URL}/rest/api/2/issue/${ISSUE_KEY}/transitions"
else
  # Update other fields
  JSON_DATA="{\"fields\": {\"$FIELD_TYPE\": \"$FIELD_VALUE\"}}"
  ENDPOINT="${JIRA_URL}/rest/api/2/issue/${ISSUE_KEY}"
fi

# Update issue
curl -s -X PUT \
  -H "Content-Type: application/json" \
  -u "${JIRA_USER}:${JIRA_API_TOKEN}" \
  --data "$JSON_DATA" \
  "$ENDPOINT"

echo "Issue $ISSUE_KEY updated successfully."
```

#### 5. Script to Add a Comment (`jira-add-comment.sh`)

```bash
#!/bin/bash

# Source authentication variables
source ./jira-auth.sh

# Get parameters
ISSUE_KEY=$1
COMMENT_TEXT=$2

if [ -z "$ISSUE_KEY" ] || [ -z "$COMMENT_TEXT" ]; then
  echo "Usage: $0  "
  exit 1
fi

# Create JSON payload
JSON_DATA="{\"body\": \"$COMMENT_TEXT\"}"

# Add comment
curl -s -X POST \
  -H "Content-Type: application/json" \
  -u "${JIRA_USER}:${JIRA_API_TOKEN}" \
  --data "$JSON_DATA" \
  "${JIRA_URL}/rest/api/2/issue/${ISSUE_KEY}/comment" | jq .
```

### Make Scripts Executable

After creating these scripts, make them executable:

```bash
chmod +x jira-*.sh
```

## Integrating with Goose

Now that we have our shell scripts ready, we can integrate them with Goose. Based on the search results, there are two primary ways to use these scripts with Goose:

### Method 1: Using the `--with-extension` Flag

You can start a Goose session with your JIRA scripts as an extension[3]:

```bash
goose session --with-extension "/path/to/jira-scripts/jira-auth.sh"
```

This approach enables Goose to interact with your JIRA scripts during a session.

### Method 2: Launching Scripts from Goose

Alternatively, you can have Goose execute your scripts on demand:

```bash
goose "Get all issues from the PROJECT project" 
```

Goose can interpret this command and execute the appropriate script:

```bash
./jira-get-issues.sh PROJECT
```

### Method 3: Create a Wrapper Extension

For a more integrated experience, create a wrapper script that Goose can use to access all your JIRA operations:

```bash
#!/bin/bash

# jira-toolkit.sh - Wrapper for JIRA operations

COMMAND=$1
shift

case $COMMAND in
  "auth")
    ./jira-auth.sh
    ;;
  "get-issues")
    ./jira-get-issues.sh "$@"
    ;;
  "create-issue")
    ./jira-create-issue.sh "$@"
    ;;
  "update-issue")
    ./jira-update-issue.sh "$@"
    ;;
  "add-comment")
    ./jira-add-comment.sh "$@"
    ;;
  *)
    echo "Unknown command: $COMMAND"
    echo "Available commands: auth, get-issues, create-issue, update-issue, add-comment"
    exit 1
    ;;
esac
```

Then start a Goose session with this wrapper:

```bash
goose session --with-extension "/path/to/jira-scripts/jira-toolkit.sh"
```

## Using Configuration Files

As an alternative to hardcoding credentials in scripts, you can use a configuration file approach similar to what's described in search result[4] for `.goosehints` files. Create a `.jira-config` file:

```
# JIRA Configuration
JIRA_URL=https://your-domain.atlassian.net
JIRA_USER=your-email@example.com
JIRA_API_TOKEN=your-api-token

# Project Settings
DEFAULT_PROJECT=PROJECT
```

Then modify your scripts to source this configuration file instead of hardcoding values.

## Complex Operations Example: Advanced JQL Queries

For more complex JIRA operations, you can create a script that handles JQL (JIRA Query Language) queries:

```bash
#!/bin/bash

# Source authentication variables
source ./jira-auth.sh

# Get JQL query from command line argument, URL encode it
JQL="$1"
ENCODED_JQL=$(echo $JQL | jq -sRr @uri)

if [ -z "$JQL" ]; then
  echo "Usage: $0 ''"
  exit 1
fi

# Execute JQL query
curl -s -X GET \
  -H "Content-Type: application/json" \
  -u "${JIRA_USER}:${JIRA_API_TOKEN}" \
  "${JIRA_URL}/rest/api/2/search?jql=${ENCODED_JQL}&maxResults=20" | \
  jq '.issues[] | {key: .key, summary: .fields.summary, status: .fields.status.name, assignee: .fields.assignee.displayName}'
```

## Conclusion

While the search results don't provide explicit instructions for JIRA integration with Goose using shell commands, we can leverage Goose's extension capabilities to create an effective integration. By creating shell scripts that interact with JIRA's REST API and then configuring Goose to use these scripts as extensions, you can automate a wide range of JIRA operations directly from your Goose workflow.

This approach combines Goose's AI capabilities with the flexibility of shell scripting to create a powerful tool for managing JIRA issues and workflows, reducing manual intervention and streamlining your development process.

Citations:
[1] https://stackoverflow.com/questions/53114197/how-to-run-shell-command-in-jira-script-runner-groovy
[2] https://block.github.io/goose/docs/guides/goose-cli-commands/
[3] https://block.github.io/goose/docs/getting-started/using-extensions/
[4] https://dev.to/lymah/using-goosehints-files-with-goose-304m
[5] https://appfire.atlassian.net/wiki/spaces/SUPPORT/pages/89141441/How+to+set+up+CLI+client+v9.0+and+above+in+Windows+and+run+CLI+commands
[6] https://community.atlassian.com/t5/Jira-Service-Management/How-to-automate-Jira-using-shell-script/qaq-p/1933849
[7] https://scottspence.com/posts/setting-up-codename-goose-in-wsl
[8] https://sourceforge.net/projects/goose-developer-agent.mirror/
[9] https://news.ycombinator.com/item?id=42879323
[10] https://github.com/block/goose/blob/main/ARCHITECTURE.md
[11] https://block.github.io/goose/docs/quickstart/
[12] https://sourceforge.net/projects/goose-developer-agent.mirror/files/v1.0.8/goose-x86_64-unknown-linux-gnu.tar.bz2/download
[13] https://github.com/block/goose/blob/main/docs/index.md
[14] https://github.com/ankitpokhrel/jira-cli
[15] https://pypi.org/project/goose-ai/0.9.7/
[16] https://marketplace.atlassian.com/apps/6398/jira-command-line-interface-cli
[17] https://docs.callgoose.com/sqibs-integration/powershell___incident_auto_remediation
[18] https://www.youtube.com/watch?v=7NmEKaX44L4
[19] https://block.xyz/open-source
[20] https://www.youtube.com/watch?v=hl5m6_0xDPU
