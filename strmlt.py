import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
load_dotenv()
import os

serper_api_key = os.getenv("SERPER_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="LinkedIn Post Writer",
    page_icon="📝",
    layout="wide"
)

st.title("✍️LinkedIn Post Writer")
st.markdown("Generate AI blog posts about the latest trends using AI agents.")

with st.sidebar:
    st.header("Content Settings")
    
    topic = "General news related to AI and Generative AI"
    
    st.markdown("### LLM Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    st.markdown("---")
    
    generate_button = st.button(
        "Generate Content", 
        type="primary", 
        use_container_width=True
    )

def generate_content(topic):
    llm = LLM(model="gemini/gemini-1.5-pro", api_key=google_api_key)
    search_tool = SerperDevTool(n=3, api_key=serper_api_key)

    # Agent 1: Senior Research Analyst
    senior_research_analyst = Agent(
        role="Senior Research Analyst",
        goal="Research, Analyze and Synthesize comprehensive information on {topic} from reliable news web sources",
        backstory="You're an expert news research analyst with advanced web research skills."
                "Research only news about AI and Generative AI of 2025."
                "You excel at finding, analyzing, and synthesizing information across the internet using search tools across various news resources."
                "You're skilled at distinguishing reliable sources from unreliable ones, fact-checking, cross-referencing information, and identifying key patterns and insights."
                "You provide well-organized research briefs with proper citations and source verification."
                "Your analysis includes both raw data and interpreted insights, making complex information accessible and actionable.",
        allow_delegation=False,
        verbose=False,
        tools=[search_tool],
        llm=llm
    )

    # Agent 2: Post Writer
    content_writer = Agent(
        role="Content Writer",
        goal="Transform research findings into engaging blog posts while maintaining accuracy",
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

    # Research Task
    research_task = Task(
        description=("""1. Conduct comprehensive research on the topic, which should cover:
                        - Recent developments and news
                        - Key industry trends and innovations
                        - Expert opinions and analysis
                        - Statistical data and market insights
                    2. Evaluate source credibility and fact-check all information
                    3. Organize information in a clear, logical structure
                    4. Include all relevant citations and sources"""),
        expected_output="""A detailed research report should contain:
                            - Executive summary of key findings
                            - Comprehensive analysis of current trends and developments
                            - List of verified facts and statistics
                            - All citations and links to original sources
                            - Clear categorization of main themes and patterns
                            Please format with clear sections and bullet points for references""",
        agent=senior_research_analyst
    )

    # Content Writing Task
    writing_task = Task(
        description=("""Using the research brief provided, create an engaging blog post that:
                        1. Transforms technical research into accessible content
                        2. Maintains all factual accuracy and citations from the research
                        3. Includes:
                            - Attention-grabbing introduction
                            - Well-structured body sections with clear headings
                            - Compelling conclusion
                        4. Preserves all source citations in [Source: URL] format
                        5. Includes a References section at the end"""),
        expected_output=""" A polished LinkedIn post in markdown format that:
                            - Engages readers while maintaining accuracy
                            - Contains properly structured sections
                            - Always sticks to one topic and doesn't jump into multiple topics.
                            - Don't include more than two URLs in the post
                            - Include 3 most relevant hashtags for the post and don't exceed the limit of the hashtags with '#' symbol before the word
                            - Present the information in an informative way
                            - Follows proper markdown formatting, using H1 for the title and try to use much headings, and then include the URL in [Source: URL] format
                            - After the main content include the hashtags keywords with '#' before them just below the main content""",
        agent=content_writer
    )

    crew = Crew(
        agents=[senior_research_analyst, content_writer],
        tasks=[research_task, writing_task],
        verbose=True
    )

    result = crew.kickoff(inputs={"topic": topic})
    # Ensure result is a string (if it's not, convert it to one)
    return str(result) if result else "No content generated."

if generate_button:
    with st.spinner('Generating content... This may take a moment.'):
        try:
            result = generate_content(topic)
            st.markdown("### Generated Content")
            st.markdown(result)
            
            # Add download button
            st.download_button(
                label="Download Content",
                data=result,
                file_name=f"{topic.lower().replace(' ', '_')}_article.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with CrewAI, Streamlit and Gemini")
