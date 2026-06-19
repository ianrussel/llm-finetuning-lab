# Help Desk Tickets Datasets

## Data Masking

Some features in the datasets were masked to hide details that are related to the dataset provider to maintain
privacy, while maintaining the consistency and relations between the datasets. The features prefixed with # were
masked.

The issue_proj feature in issues.csv and issues_snapshot.csv defines an issue related project. It is defined by the
helpdesk team for internal and external projects. For the external projects, it is more than six characters that
follows a pattern of having two characters as the country code, a product key, and a 3+ characters client code.
For example, ZZ99ABC is key for product 99 for client ABC in country ZZ. 
It was found there were 18 unique countries codes for the external projects, and those codes were mapped while masking
the feature with C00 to C17, so in the previous example, if ZZ was mapped to C03, then the project code will be
C0378XPS, assuming that 99ABC was masked to 78XPS. 

## DataSets

### issues.csv

This dataset is a CSV file containing issues reported through the helpdesk system from January 2016 until
March 2023. Once an issue is reported by the customer, it walks through a workflow until it is resolved, noting that
some steps could be passed multiple times. Image [process-flow](./process-flow.png) describes the flow.

The below are the features:

| **Feature**           | **Type**    | **Description**                                                       |
|-----------------------|-------------|-----------------------------------------------------------------------|
| id                    | numeric     | The unique ID of the issue.                                           |
| issue_num             | numeric     | The unique sequence of the issue within the project.                  |
| # issue_proj          | text        | The issue's project code.                                             |
| # issue_reporter      | text        | The user ID who reported the issue.                                   |
| # issue_assignee      | text        | The last user ID assigned to work on  the issue.                      |
| issue_contr_count     | numeric     | The total number of users worked on this issue.                       |
| issue_priority        | categorical | The priority of the issue.                                            |
| issue_type            | categorical | The nature of the issue, like incident or service request.            |
| issue_created         | timestamp   | The creation time of the issue.                                       |
| issue_resolution_date | timestamp   | The resolution time of the issue.                                     |
| issue_resolution      | text        | One of the values: Done, Won't Do, Cannot Reproduce, or Duplicate.    |
| issue_status          | text        | The issue last status value: Done, In Progress, etc...                |
| issue_comments_count  | numeric     | The number of the issue comments.                                     |
| last_change_date      | timestamp   | The last time the issue was updated.                                  |
| wf_total_time         | numeric     | The total processing time for the issue (in seconds).                 |
| processing_steps      | numeric     | The number of steps the issue took until it reached its final status. |
|                       |             | **Repeated features for each step "S"**                               |
| wf_{S}                | numeric     | The time spent while the issue was in the "S" state.                  |
| wfe_{S}               | numeric     | The number of times the issue passed state "S".                       |

The wf_ and wfe_ features are the result of aggregations and calculations collected from the issues_change_history.csv.
The workflow steps that were found in the database are below, sorted by the mostly passed steps to the least:

- Open
- In progress
- Resolved
- Waiting
- Validation
- Resolved under monitoring
- Pending deployment
- Closed
- Reopened
- Cancelled
- In review
- To do
- Under review
- Rejected
- Done
- Approved

### issues_snapshot.csv

An issue could be handled by multiple support engineers, which requires to track each member activity separately.
This feature captures each assignee activity, which is gathered and calculated from the issues_change_history.csv
dataset. This dataset contains the same features as in issues.csv with the additional below features:

| **Feature** | **Type** | **Description**                                             |
|-------------|----------|-------------------------------------------------------------|
| started     | numeric  | When the assignee started to work on this issue.            |
| ended       | numeric  | When the assignee completed his work on this ticket.        |
| turn        | text     | The assignee turn sequence when was assigned to this issue. |

### sample_utterances.csv

The messages exchanged between the helpdesk team and the customer for all the issues in issues_snapshot_sample.xlxs.
The messages are broken into utterances (sentences).

| **Feature** | **Type**  | **Description**                                                                                                                                  |
|-------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| issuedid    | numeric   | The utterance's issue ID.                                                                                                                        |
| id          | numeric   | The comment ID this utterance was extracted from.                                                                                                |
| comment_seq | numeric   | The sequence of this comment in the related issue. Starting from 0.                                                                              | 
| utr_seq     | numeric   | The sequence of this utterance in the comment text. Starting from 0.                                                                             |
| # author    | text      | The author of this utterance, which will be the same as the author of the related comment that could be the reporter (customer) or the assignee. |
| created     | timestamp | The related comment creation time.                                                                                                               |
| is_private  | numeric   | A flag which determines if the related comment was a private comment or not.                                                                     |
| actionbody  | text      | The utterance text.                                                                                                                              |
| author_role | text      | The role of this utterance author while working on the related ticket, which could be as: reporter, assignee, or others.                         |

Some words and sentences where masked to maintain data provider privacy, and to provide common terms (placeholders) for
similar contexts. For example: if a script was provided in a comment to fix an issue on site, then what is important that
a script was provided regardless of its content.
The list below shows each placeholder and what it replaces:

| Placeholder     | What represents                                                                                                               |
|-----------------|-------------------------------------------------------------------------------------------------------------------------------|
| ph_code         | A code that was provided in a comment.                                                                                        |
| ph_email_thread | An email body that was written in a comment.                                                                                  |
| ph_file         | Mentioning a file that was attached to an issue or a file that is a part of a technical discussion.                           |
| ph_info_table   | A tabular data, similar like markdown tables, that holds technical information.                                               |
| ph_ip_address   | An IP address or a domain mentioned in the ticket.                                                                            |
| ph_item         | An item type that exists in a system. For example: if the system was a banking system, it could be an account or transaction  |
| ph_link         | A link, such as meeting invites or zoom meeting links                                                                         |
| ph_log          | System logs that was attached to the comments.                                                                                |
| ph_name         | Mentioning persons, systems, products, or any similar entity.                                                                 |
| ph_path         | File paths that were mentioned in the comment, usually as part of a technical procedure.                                      |
| ph_sql          | SQL scripts and queries that were added to the comments, usually to trace issues or retrieve data.                            |
| ph_technical    | Any technical words used in the comments, like application pages, scripts, component names, sample data, customer data, etc.. |
| ph_url          | Mentioning URLs.                                                                                                              |
| ph_user         | When a user tags another user in a comment, which is usually used to push notification or to highlight an important note.     |
| ph_version      | Version numbers related to the company products.                                                                              |

### issues_snapshot_sample.xlsx

A stratified sample was taken from the issues.csv and then handed to the helpdesk manager to score the performance for
each assignee on each issue. The helpdesk manager scored the performance against the three targets below:

- Q1: (Quality of Work): Work is accurate and precise.
- Q2: (Quality of Work): Displays thoroughness and completeness in work activity.
- Q3: (Client Relations): Responsive and courteous to client inquiries.

| **Column**     | **Description**                                                                          |
|----------------|------------------------------------------------------------------------------------------|
| id             | The issue ID                                                                             |
| no             | The issue unique No. within the project                                                  |
| # project      | The issue Project                                                                        |
| # reporter     | The issue Reporter                                                                       |
| type           | The issue type                                                                           |
| priority       | The issue priority                                                                       |
| contributors   | The total number of contributors                                                         |
| turn_no        | The turn sequence for this record assignee                                               |
| # assignee     | The turn assignee                                                                        |
| started        | When the issue was assigned to 'assignee'                                                |
| ended          | When the 'assignee' finished his turn                                                    |
| spent hours    | The total hours spent by the 'assignee' on this issue                                    |
| steps          | The total number of steps done by the 'assignee'                                         |
| comments count | The total number of comments added to the ticket while it was assigned to the 'assignee' |
| valid          | A flag to mark if this record could be appraised or not                                  |
| Q1             | The score of Q1                                                                          |
| Q2             | The score of Q2                                                                          |
| Q3             | The score of Q3                                                                          |
| Notes          | Notes added by the annotator explaining his scores if needed                             |

### issues_change_history.csv

This dataset contains the history of changes made to the issues assignee and workflow processing status.

| **Feature**     | **Type**  | **Description**                                                                                                   |
|-----------------|-----------|-------------------------------------------------------------------------------------------------------------------|
| id              | numeric   | Change history record unique id.                                                                                  |
| issueid         | numeric   | The related issueid.                                                                                              |
| field           | text      | The changed field. In this dataset it is "assignee" or "status".                                                  |
| # value         | text      | The new value after the change. The value was masked only if the changed field was "assignee"                     |
| created         | timestamp | When this change was created.                                                                                     |
| change_group_id | numeric   | Multiple changes could be done in one single update to the issue, and those changes are grouped using this field. |