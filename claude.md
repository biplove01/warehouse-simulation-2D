# System Overview
- The system is a warehouse simulation system. We've used reinforcement Learning in order to make the agents in the warehouse autonomous.
- Word.py is our environment here whereas robot.py is the agent. We've also successfully trained our multiagent robot.
- The world consists of a start station in warehouse where there are 5 tiles which are dedicated for robots to sibt at
- There are obstacles and shelves in warehouse
## Agent States and Movements
- The agent has 5 movments ie up,down,left right and no movement
- The start state is when the agents are there in their initial position
- The second state is when the robots find the path from the initial position to the particular shelve to pick up the package
- The thrid state is when the agent reach the desination of shelves they are handed over package that they pick up now they find the path from shelves to delivery statation.
- The final stattion is after the agents have completed the delivery, they go and find the path from delivery station to start state
- The agents should be in rest where there is no any order

# Intended System Flow
- Order is placed in ecommerce System
- THe order is processed and published to Kafka
- The pygame reads it(reader.py) is how the pygame reads it and sends it to the internal queue(queuemanager.py)
- Then from each order the list of items that are ordered are read.
- Further from each of the items the quantity is also read.
- The queue manager assigs each of the agent with each item with ordertracercode.
- The agent should pick up the item based on the size and itemcode of the  item
- The agents should go from pickup station to the dedicated shelve and from there to goal station exactly quantity times.
- After each of the item is delivered by the given agent a http request should be made at http://localhost:8080/api/v1/admin/pack/item/{ordertracercode} with http header named SystemKey and value SystemKey illsontpygamesystem.

# What I want you to do
- The dual agent works but the logic for integration is quite broken. The intended flow is not achieved.
- First I want to let you know there are 10 items and 5 sizes(Small,Medium,Large,XL,XXL)
- in folder randSim there is intended flow how the system should be able to handle indexing
- Here the indexing is the process of assiging the given shelve a intended coordinate as per the item
- I dont know about the grid world environment much I want you to map item and size to a fixed shelve in the grid world. There are 10 items in ecommerce website whose item code range from 1 to 10 and further 5 intended sizes. So you should be able to map it right.
- further please fix the broken logic of dual agent integration. the reader.py reads from the topic and get the string which is mapped according. Make the queue and then make it follow the intended system flow pls.

### Example Usage
- User Orders with oID : 300
- The list consists of 3 items with itemCode 2 Small,8 Large,4 XL quantity 2,1,5 with ordertracer code of 432,5444,2132
- Agent 1 is alloted for handling itemCode 2, Agent 2 is for itemCode 8
- Shelve number is calculated based on itemcode and size
- Agent 1 is at rest at 0,0. The shelve coridinate is mapped to (40,23) so the agent 1 finds the path to that shelve with pretrained values. After the picking up it finds the path to delivary station (100,0) and returns to start position. Agent 1 does this twice(quantity times) and after a orderedItem is handled by it it sends api request to http://localhost:8080/api/v1/admin/pack/item/432. 
- Similarly Agent 2 does the same thing. Since Agent 2 has only 1 item with 1 quantity to deliver it completes the task first and hence it is handed over for itemcode 5 as well. 
- When the queue is completed the agents rests at their respective start position.