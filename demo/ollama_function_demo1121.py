import os
import ollama
import asyncio


# Simulates an API call to get flight times
# In a real application, this would fetch data from a live database or API
def write_file(path: str, content: str) -> str:
  result = '写入完成!';
  try:
    directory_path = os.path.dirname(path)
    if not os.path.exists(directory_path):
      os.makedirs(directory_path)
    with open(path, 'w') as f:
      f.write(content)
  except (FileExistsError, OSError) as e:
    print(f"目录创建失败，错误信息: {e}")
    result = f"目录创建失败，错误信息: {e}"
  except OSError as e:
    print(f"文件写入失败，错误信息: {e}")
    result = f"文件写入失败，错误信息: {e}"
  return result;


async def run(model: str):
  client = ollama.AsyncClient(host='localhost')
  # Initialize conversation with a user query
  messages = [{'role': 'user', 'content': '帮我写一篇关于周恩来的人物介绍3000字左右到本地电脑D:/周恩来.txt文件中'}]

  # First API call: Send the query and function description to the model
  response = await client.chat(
    model=model,
    messages=messages,
    tools=[
      {
        'type': 'function',
        'function': {
          'name': 'write_file',
          'description': '写文件到本地磁盘',
          'parameters': {
            'type': 'object',
            'properties': {
              'path': {
                'type': 'string',
                'description': '文件路径，用于创建文件',
              },
              'content': {
                'type': 'string',
                'description': '文件内容，用于写入指定文件',
              },
            },
            'required': ['path', 'content'],
          },
        },
      },
    ],
  )

  # Add the model's response to the conversation history
  messages.append(response['message'])

  # Check if the model decided to use the provided function
  if not response['message'].get('tool_calls'):
    print("The model didn't use the function. Its response was:")
    print(response['message']['content'])
    return

  # Process function calls made by the model
  if response['message'].get('tool_calls'):
    available_functions = {
      'write_file': write_file,
    }
    for tool in response['message']['tool_calls']:
      function_to_call = available_functions[tool['function']['name']]
      function_response = function_to_call(tool['function']['arguments']['path'], tool['function']['arguments']['content'])
      # Add function response to the conversation
      messages.append(
        {
          'role': 'tool',
          'content': function_response,
        }
      )

  # Second API call: Get final response from the model
  final_response = await client.chat(model=model, messages=messages)
  print(final_response['message']['content'])


# Run the async function
asyncio.run(run('llama3.2'))