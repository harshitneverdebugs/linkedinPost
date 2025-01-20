from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
load_dotenv()
import os
import litellm

groq_api_key = os.getenv("GROQ_API_KEY")  
serper_api_key = os.getenv("SERPER_API_KEY")

topic="General news related to AI and Generative AI"

llm=LLM(model="groq/llama-3.3-70b-versatile", api_key=groq_api_key, max_tokens=800)

search_tool=SerperDevTool(n=3, api_key=serper_api_key)
os.environ['LITELLM_LOG'] = 'DEBUG'
litellm.set_verbose=True

#Agent 1
senior_research_analyst=Agent(
    role="Senior Research Analsyt",
    goal="Research, Analyze and Synthesize comprehensive information on {topic} from reliable web sources",
    backstory="You're an expert research analyst with advanced web research skills."
              "Reasearch only news about AI and Generative AI of 2025"
              "You excel at finding, analyzing, and synthesizing information across the internet using search tools."
              "You're skilled at distinguishing reliable sources from unreliable ones, fact-checking, cross-referencing information, and identifying key patterns and insights."
              "You provide well-organized research briefs with proper citations and source verification."
              "Your analysis includes both raw data and interpreted insights, making complex information accessible and actionable.",
    allow_delegation=False,
    verbose=False,
    tools=[search_tool],
    llm=llm
)

#Agent 2
content_writer=Agent(
    role="Content Writer",
    goal="Transform reaserch findings into engaging blog posts while maintaining accuracy",
    backstory="You're a skilled content writer specialized in creating"
              "engaging, accessible content from technical research."
              "You work closely with the Senior Research Analyst and excel at maintaining the perfect"
              "balance between informative and entertaining writing,"
              "while ensuring all facts and citations from the research"
              "are properly incorporated. You have a talent for making"
              "complex topics approachable without oversimplifying them.",
    allow_delegation=False,
    verbose=False,
    llm=llm
)

#Research Task
resarch_tasks=Task(
    description=("""
        1.Conduct comprehensive research on the topic, which should cover:
            -Recent developments and news
            -Key industry trends and innovations
            -Expert opinions and analysis
            -Statistical data and market insights
        2.Evaluate source credibility and fact-check all information
        3.Organize information in a clear, logical structure
        4.Include all relevant citations and sources
        """),
    expected_output=""""A detailed research report should contain:
                -Executive summary of key findings
                -Comprehensive analysis of current trends and developments
                -List of verified facts and statistics
                -All citations and links to original sources
                -Clear categorization of main themes and patterns
                Please format with clear sections and bullet points for refences""",
    agent=senior_research_analyst
)

#Task 2 Content Writing
writing_task=Task(
    description=("""Using the research brief provided, create an engaging blog post that:
                    1.Transforms technical research into accessible content
                    2.Maintains all factual accuracy and citations from the research
                    3.Includes:
                        -Attention-grabbing introduction
                        -Well-structured body sections with clear headings
                        -Compelling conclusion
                    4.Preserves all source citations in [Source: URL] format
                    5.Includes a References section at the end
                        """),
    expected_output=""" A polished LinkedIn post in markdown format that:
        -Engages readers while maintaining accuracy
        -Contains properly structured sections
        -Includes inline citations hyperlinked to the original source URL but not include more than 2 URL in the post
        -Include 3 most relevant hashtags for the post and don't exceed the limit of the hashtags with '#' symbol before the word 
        -Presents information in an informative way
        -The post overall character limit should not exceed more than 3000 characters
        -Follows proper markdown formatting, using H1 for the title and H3 for the sub-sections, after the main content include the hashtags keywords with'#' before them just below the sub-sections and and then include the URL in [Source: URL] format""",
    agent=content_writer
)

crew=Crew(
    agents=[senior_research_analyst, content_writer],
    tasks=[resarch_tasks, writing_task],
    verbose=True
)

result=crew.kickoff(inputs={"topic":topic})
print(result)