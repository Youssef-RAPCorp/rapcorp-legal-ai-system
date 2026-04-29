"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    LEGAL KNOWLEDGE SWARM                                      ║
║              RAPCorp Legal AI System                                          ║
║                                                                               ║
║  15 Specialized Mini-Agents for Parallel Legal Research:                      ║
║  • Case Law Retriever       • Statute Finder        • Contract Parser         ║
║  • Precedent Analyzer       • Citation Hunter       • Jurisdiction Tracker    ║
║  • Contradiction Finder     • Gap Identifier        • Timeline Builder        ║
║  • Regulatory Monitor       • Argument Mapper       • Risk Assessor           ║
║  • Damages Calculator       • Deadline Tracker      • Brief Synthesizer       ║
║                                                                               ║
║  Features:                                                                    ║
║  • Parallel execution via asyncio                                             ║
║  • Communication hub for agent coordination                                   ║
║  • State-aware retrieval                                                      ║
║  • Cost-optimized model selection                                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable, Set
from enum import Enum
import json
import uuid

from configs.config import LegalAIConfig, USState, LegalDomain, GeminiModel


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class AgentType(Enum):
    """Types of legal research agents."""
    CASE_LAW_RETRIEVER = "case_law_retriever"
    STATUTE_FINDER = "statute_finder"
    CONTRACT_PARSER = "contract_parser"
    PRECEDENT_ANALYZER = "precedent_analyzer"
    CITATION_HUNTER = "citation_hunter"
    JURISDICTION_TRACKER = "jurisdiction_tracker"
    CONTRADICTION_FINDER = "contradiction_finder"
    GAP_IDENTIFIER = "gap_identifier"
    TIMELINE_BUILDER = "timeline_builder"
    REGULATORY_MONITOR = "regulatory_monitor"
    ARGUMENT_MAPPER = "argument_mapper"
    RISK_ASSESSOR = "risk_assessor"
    DAMAGES_CALCULATOR = "damages_calculator"
    DEADLINE_TRACKER = "deadline_tracker"
    BRIEF_SYNTHESIZER = "brief_synthesizer"


class AgentStatus(Enum):
    """Status of an agent."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentMessage:
    """Message passed between agents via the communication hub."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: AgentType = None
    recipient: Optional[AgentType] = None  # None = broadcast
    message_type: str = "info"
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 5  # 1-10, lower = higher priority
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender.value if self.sender else None,
            "recipient": self.recipient.value if self.recipient else None,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority
        }


@dataclass
class AgentResult:
    """Result from an agent's work."""
    agent_type: AgentType
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# COMMUNICATION HUB
# ═══════════════════════════════════════════════════════════════════════════════

class CommunicationHub:
    """
    Central hub for agent-to-agent communication.
    
    Agents can:
    - Send messages to specific agents
    - Broadcast messages to all agents
    - Subscribe to message types
    - Query shared state
    """
    
    def __init__(self):
        self._messages: List[AgentMessage] = []
        self._subscribers: Dict[str, List[Callable]] = {}
        self._shared_state: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message through the hub."""
        async with self._lock:
            self._messages.append(message)
        
        # Notify subscribers
        message_type = message.message_type
        if message_type in self._subscribers:
            for callback in self._subscribers[message_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)
                except Exception as e:
                    print(f"Warning: Subscriber callback failed: {e}")
    
    async def broadcast(
        self,
        sender: AgentType,
        message_type: str,
        content: Dict[str, Any],
        priority: int = 5
    ) -> None:
        """Broadcast a message to all agents."""
        message = AgentMessage(
            sender=sender,
            recipient=None,
            message_type=message_type,
            content=content,
            priority=priority
        )
        await self.send_message(message)
    
    def subscribe(self, message_type: str, callback: Callable) -> None:
        """Subscribe to a message type."""
        if message_type not in self._subscribers:
            self._subscribers[message_type] = []
        self._subscribers[message_type].append(callback)
    
    def unsubscribe(self, message_type: str, callback: Callable) -> None:
        """Unsubscribe from a message type."""
        if message_type in self._subscribers:
            self._subscribers[message_type] = [
                cb for cb in self._subscribers[message_type] 
                if cb != callback
            ]
    
    async def get_messages_for(
        self,
        recipient: AgentType,
        since: Optional[datetime] = None
    ) -> List[AgentMessage]:
        """Get messages for a specific agent."""
        async with self._lock:
            return [
                m for m in self._messages
                if (m.recipient is None or m.recipient == recipient)
                and (since is None or m.timestamp > since)
            ]
    
    async def set_shared_state(self, key: str, value: Any) -> None:
        """Set shared state that all agents can access."""
        async with self._lock:
            self._shared_state[key] = value
    
    async def get_shared_state(self, key: str, default: Any = None) -> Any:
        """Get shared state."""
        async with self._lock:
            return self._shared_state.get(key, default)
    
    async def update_shared_state(self, key: str, update_fn: Callable) -> Any:
        """Atomically update shared state."""
        async with self._lock:
            current = self._shared_state.get(key)
            new_value = update_fn(current)
            self._shared_state[key] = new_value
            return new_value
    
    def get_message_count(self) -> int:
        """Get total message count."""
        return len(self._messages)


# ═══════════════════════════════════════════════════════════════════════════════
# BASE AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """
    Base class for all legal research agents.
    
    Each agent:
    - Has a specific type and purpose
    - Can communicate via the hub
    - Runs asynchronously
    - Reports costs and metrics
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        config: LegalAIConfig,
        hub: CommunicationHub,
        llm_client: Any = None  # GeminiClient
    ):
        self.agent_type = agent_type
        self.config = config
        self.hub = hub
        self.llm = llm_client
        self.status = AgentStatus.IDLE
        self._last_result: Optional[AgentResult] = None
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what this agent does."""
        pass
    
    @property
    def preferred_model(self) -> str:
        """The preferred Gemini model for this agent's tasks."""
        return GeminiModel.FLASH.value  # Default to Flash
    
    @abstractmethod
    async def process(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> AgentResult:
        """
        Process a query and return results.
        
        Args:
            query: The research query
            context: Additional context (state, domain, case_id, etc.)
        
        Returns:
            AgentResult with findings
        """
        pass
    
    async def run(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> AgentResult:
        """Run the agent with status tracking."""
        self.status = AgentStatus.RUNNING
        start_time = datetime.utcnow()
        
        try:
            # Announce start
            await self.hub.broadcast(
                self.agent_type,
                "agent_started",
                {"query": query[:100]}
            )
            
            # Process
            result = await self.process(query, context)
            
            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds()
            result.duration_seconds = duration
            
            self.status = AgentStatus.COMPLETED
            self._last_result = result
            
            # Announce completion
            await self.hub.broadcast(
                self.agent_type,
                "agent_completed",
                {
                    "success": result.success,
                    "duration": duration,
                    "summary": self._summarize_result(result)
                }
            )
            
            return result
            
        except Exception as e:
            self.status = AgentStatus.FAILED
            result = AgentResult(
                agent_type=self.agent_type,
                success=False,
                error=str(e)
            )
            self._last_result = result
            
            await self.hub.broadcast(
                self.agent_type,
                "agent_failed",
                {"error": str(e)}
            )
            
            return result
    
    def _summarize_result(self, result: AgentResult) -> str:
        """Create a brief summary of the result for broadcasting."""
        if not result.success:
            return f"Failed: {result.error}"
        
        data = result.data
        if "count" in data:
            return f"Found {data['count']} items"
        if "findings" in data:
            return f"{len(data['findings'])} findings"
        return "Completed"
    
    async def _call_llm(
        self,
        prompt: str,
        task: Optional[str] = None,
        system_instruction: Optional[str] = None,
        return_json: bool = False
    ) -> Dict[str, Any]:
        """Helper to call the LLM with proper error handling."""
        if not self.llm:
            raise RuntimeError(f"No LLM client configured for {self.agent_type.value}")
        
        return await self.llm.generate(
            prompt=prompt,
            task=task,
            model_override=self.preferred_model,
            system_instruction=system_instruction,
            return_json=return_json
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIALIZED AGENTS
# ═══════════════════════════════════════════════════════════════════════════════

class CaseLawRetriever(BaseAgent):
    """
    Retrieves relevant case law using CourtListener (real API) when available,
    falls back to LLM-generated search strategy when no API key is configured.
    """

    def __init__(
        self,
        config: LegalAIConfig,
        hub: CommunicationHub,
        llm_client: Any = None,
        courtlistener_client: Any = None,   # CourtListenerClient instance or None
    ):
        super().__init__(AgentType.CASE_LAW_RETRIEVER, config, hub, llm_client)
        self.courtlistener = courtlistener_client

    @property
    def description(self) -> str:
        source = "CourtListener API" if self.courtlistener else "LLM strategy"
        return f"Retrieves relevant case law and precedents ({source})"

    @property
    def preferred_model(self) -> str:
        return GeminiModel.FLASH.value

    async def process(self, query: str, context: Dict[str, Any]) -> AgentResult:
        state = context.get("state", USState.FEDERAL)
        domain = context.get("domain", LegalDomain.CONTRACT)
        total_cost = 0.0
        total_tokens = 0

        # ── Step 1: Use LLM to extract focused search terms ────────────────
        search_strategy: Dict[str, Any] = {}
        search_terms: list = [query]   # fallback: use raw query

        if self.llm:
            strategy_prompt = f"""Extract the best legal search terms for a CourtListener case law search.

Legal question: {query}
Jurisdiction: {state.value}
Domain: {domain.value}

Return a JSON object:
{{
    "primary_query": "the best single search string for full-text case law search",
    "search_terms": ["term1", "term2", "term3"],
    "case_types": ["supreme_court", "appeals", "district"],
    "date_range": {{"start": "YYYY", "end": "YYYY"}},
    "relevance_factors": ["factor1", "factor2"]
}}"""
            llm_result = await self._call_llm(strategy_prompt, task="retrieval", return_json=True)
            total_cost += llm_result.get("cost", 0)
            total_tokens += llm_result.get("input_tokens", 0) + llm_result.get("output_tokens", 0)

            try:
                search_strategy = json.loads(llm_result.get("text", "{}"))
                primary_query = search_strategy.get("primary_query", "").strip()
                if primary_query:
                    search_terms = [primary_query]
            except (json.JSONDecodeError, TypeError):
                pass

        # ── Step 2: Hit CourtListener if configured ─────────────────────────
        cases_data: list = []
        api_status = "not_configured"

        if self.courtlistener and self.courtlistener.is_configured:
            jurisdiction = state.value if state != USState.FEDERAL else None
            cl_result = await self.courtlistener.search_opinions(
                query=search_terms[0],
                jurisdiction=jurisdiction,
                page_size=10,
            )

            if cl_result.api_available:
                api_status = "ok"
                for case in cl_result.cases:
                    cases_data.append({
                        "case_name": case.case_name,
                        "citation": case.citation,
                        "court": case.court,
                        "date_filed": case.date_filed,
                        "url": case.url,
                        "snippet": case.snippet[:300] if case.snippet else "",
                        "relevance_score": case.relevance_score,
                        "cluster_id": case.cluster_id,
                        "docket_number": case.docket_number,
                    })
            else:
                api_status = f"error: {cl_result.error}"

        return AgentResult(
            agent_type=self.agent_type,
            success=True,
            data={
                "search_strategy": search_strategy,
                "cases": cases_data,
                "count": len(cases_data),
                "jurisdiction": state.value,
                "domain": domain.value,
                "api_status": api_status,
                "source": "courtlistener" if cases_data else "llm_strategy_only",
            },
            tokens_used=total_tokens,
            cost=total_cost,
        )


class StatuteFinder(BaseAgent):
    """Finds applicable statutes and regulations."""
    
    def __init__(self, config: LegalAIConfig, hub: CommunicationHub, llm_client: Any = None):
        super().__init__(AgentType.STATUTE_FINDER, config, hub, llm_client)
    
    @property
    def description(self) -> str:
        return "Locates applicable statutes, codes, and regulations"
    
    @property
    def preferred_model(self) -> str:
        return GeminiModel.FLASH.value
    
    async def process(self, query: str, context: Dict[str, Any]) -> AgentResult:
        state = context.get("state", USState.FEDERAL)
        
        search_prompt = f"""
        Identify relevant statutes for this legal question:
        
        Query: {query}
        Jurisdiction: {state.value}
        
        Return a JSON object with:
        {{
            "federal_statutes": ["USC citation1", "USC citation2"],
            "state_statutes": ["state citation1", "state citation2"],
            "regulations": ["CFR citation1"],
            "relevance_explanation": "brief explanation"
        }}
        """
        
        if self.llm:
            result = await self._call_llm(
                search_prompt,
                task="retrieval",
                return_json=True
            )
            
            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                data={
                    "statutes": result.get("text", "{}"),
                    "jurisdiction": state.value
                },
                tokens_used=result.get("input_tokens", 0) + result.get("output_tokens", 0),
                cost=result.get("cost", 0)
            )
        
        return AgentResult(
            agent_type=self.agent_type,
            success=True,
            data={"statutes": [], "jurisdiction": state.value}
        )


class PrecedentAnalyzer(BaseAgent):
    """Analyzes precedents and their applicability."""
    
    def __init__(self, config: LegalAIConfig, hub: CommunicationHub, llm_client: Any = None):
        super().__init__(AgentType.PRECEDENT_ANALYZER, config, hub, llm_client)
    
    @property
    def description(self) -> str:
        return "Analyzes case precedents and determines their applicability"
    
    @property
    def preferred_model(self) -> str:
        return GeminiModel.PRO.value  # Needs deeper reasoning
    
    async def process(self, query: str, context: Dict[str, Any]) -> AgentResult:
        cases = context.get("cases", [])
        
        top_cases = cases[:5]  # Limit for context
        analysis_prompt = f"""
        Analyze these precedents for the legal question:

        Question: {query}
        Cases: {json.dumps(top_cases)}

        For each case, determine:
        1. Is it binding or persuasive?
        2. Can it be distinguished?
        3. What's its holding?
        4. How does it apply to our facts?
        
        Return analysis in JSON format.
        """
        
        if self.llm:
            result = await self._call_llm(
                analysis_prompt,
                task="irac_analysis",
                return_json=True
            )
            
            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                data={
                    "analysis": result.get("text", "{}"),
                    "cases_analyzed": len(cases)
                },
                tokens_used=result.get("input_tokens", 0) + result.get("output_tokens", 0),
                cost=result.get("cost", 0)
            )
        
        return AgentResult(
            agent_type=self.agent_type,
            success=True,
            data={"analysis": {}, "cases_analyzed": 0}
        )


class ContradictionFinder(BaseAgent):
    """Identifies contradictions in legal arguments or sources."""
    
    def __init__(self, config: LegalAIConfig, hub: CommunicationHub, llm_client: Any = None):
        super().__init__(AgentType.CONTRADICTION_FINDER, config, hub, llm_client)
    
    @property
    def description(self) -> str:
        return "Identifies contradictions and conflicts in legal sources"
    
    @property
    def preferred_model(self) -> str:
        return GeminiModel.PRO.value
    
    async def process(self, query: str, context: Dict[str, Any]) -> AgentResult:
        sources = context.get("sources", [])
        
        prompt = f"""
        Identify any contradictions or conflicts in these legal sources:
        
        Query Context: {query}
        Sources: {json.dumps(sources[:10])}
        
        Look for:
        1. Conflicting holdings between cases
        2. Statutory conflicts
        3. Circuit splits
        4. Overruled precedents
        
        Return findings in JSON format.
        """
        
        if self.llm:
            result = await self._call_llm(prompt, task="legal_synthesis", return_json=True)
            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                data={"contradictions": result.get("text", "[]")},
                tokens_used=result.get("input_tokens", 0) + result.get("output_tokens", 0),
                cost=result.get("cost", 0)
            )
        
        return AgentResult(
            agent_type=self.agent_type,
            success=True,
            data={"contradictions": []}
        )


class ArgumentMapper(BaseAgent):
    """Maps argument structures and relationships."""
    
    def __init__(self, config: LegalAIConfig, hub: CommunicationHub, llm_client: Any = None):
        super().__init__(AgentType.ARGUMENT_MAPPER, config, hub, llm_client)
    
    @property
    def description(self) -> str:
        return "Maps legal argument structures and identifies strengths/weaknesses"
    
    @property
    def preferred_model(self) -> str:
        return GeminiModel.PRO.value
    
    async def process(self, query: str, context: Dict[str, Any]) -> AgentResult:
        position = context.get("position", "plaintiff")
        
        prompt = f"""
        Map the argument structure for this legal position:
        
        Issue: {query}
        Position: {position}
        
        Create an argument map with:
        1. Main claims
        2. Supporting arguments
        3. Evidence needed
        4. Potential counterarguments
        5. Rebuttals
        
        Return in JSON format.
        """
        
        if self.llm:
            result = await self._call_llm(prompt, task="argument_construction", return_json=True)
            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                data={"argument_map": result.get("text", "{}")},
                tokens_used=result.get("input_tokens", 0) + result.get("output_tokens", 0),
                cost=result.get("cost", 0)
            )
        
        return AgentResult(
            agent_type=self.agent_type,
            success=True,
            data={"argument_map": {}}
        )


class RiskAssessor(BaseAgent):
    """Assesses legal risks and outcomes."""
    
    def __init__(self, config: LegalAIConfig, hub: CommunicationHub, llm_client: Any = None):
        super().__init__(AgentType.RISK_ASSESSOR, config, hub, llm_client)
    
    @property
    def description(self) -> str:
        return "Assesses legal risks and likelihood of outcomes"
    
    @property
    def preferred_model(self) -> str:
        return GeminiModel.PRO.value
    
    async def process(self, query: str, context: Dict[str, Any]) -> AgentResult:
        facts = context.get("facts", {})
        
        prompt = f"""
        Assess the legal risks for this matter:
        
        Issue: {query}
        Facts: {json.dumps(facts)}
        
        Evaluate:
        1. Likelihood of success (1-10)
        2. Key risk factors
        3. Mitigating factors
        4. Best/worst case scenarios
        5. Recommended actions
        
        Return assessment in JSON format.
        """
        
        if self.llm:
            result = await self._call_llm(prompt, task="risk_assessment", return_json=True)
            return AgentResult(
                agent_type=self.agent_type,
                success=True,
                data={"risk_assessment": result.get("text", "{}")},
                tokens_used=result.get("input_tokens", 0) + result.get("output_tokens", 0),
                cost=result.get("cost", 0)
            )
        
        return AgentResult(
            agent_type=self.agent_type,
            success=True,
            data={"risk_assessment": {}}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE SWARM
# ═══════════════════════════════════════════════════════════════════════════════

class LegalKnowledgeSwarm:
    """
    Orchestrates multiple specialized agents for parallel legal research.
    
    Modes:
    - QUICK: 3-5 agents for fast answers
    - STANDARD: 8-10 agents for balanced research
    - COMPREHENSIVE: All 15 agents for deep analysis
    """
    
    def __init__(
        self,
        config: LegalAIConfig,
        llm_client: Any = None,
        courtlistener_client: Any = None,   # CourtListenerClient or None
    ):
        self.config = config
        self.llm = llm_client
        self.hub = CommunicationHub()

        # Initialize all agents
        self.agents: Dict[AgentType, BaseAgent] = {
            AgentType.CASE_LAW_RETRIEVER: CaseLawRetriever(
                config, self.hub, llm_client, courtlistener_client
            ),
            AgentType.STATUTE_FINDER: StatuteFinder(config, self.hub, llm_client),
            AgentType.PRECEDENT_ANALYZER: PrecedentAnalyzer(config, self.hub, llm_client),
            AgentType.CONTRADICTION_FINDER: ContradictionFinder(config, self.hub, llm_client),
            AgentType.ARGUMENT_MAPPER: ArgumentMapper(config, self.hub, llm_client),
            AgentType.RISK_ASSESSOR: RiskAssessor(config, self.hub, llm_client),
            # Additional agents can be added here...
        }
    
    def get_agents_for_mode(self, mode: str) -> List[AgentType]:
        """Get the agents to use for a research mode."""
        if mode == "quick":
            return [
                AgentType.CASE_LAW_RETRIEVER,
                AgentType.STATUTE_FINDER,
                AgentType.RISK_ASSESSOR
            ]
        elif mode == "standard":
            return [
                AgentType.CASE_LAW_RETRIEVER,
                AgentType.STATUTE_FINDER,
                AgentType.PRECEDENT_ANALYZER,
                AgentType.ARGUMENT_MAPPER,
                AgentType.RISK_ASSESSOR
            ]
        else:  # comprehensive
            return list(self.agents.keys())
    
    async def research(
        self,
        query: str,
        context: Dict[str, Any],
        mode: str = "standard"
    ) -> Dict[str, Any]:
        """
        Run parallel research with multiple agents.
        
        Args:
            query: The legal research question
            context: Context including state, domain, facts, etc.
            mode: Research mode (quick/standard/comprehensive)
        
        Returns:
            Aggregated results from all agents
        """
        agent_types = self.get_agents_for_mode(mode)
        
        # Create tasks for parallel execution
        tasks = []
        for agent_type in agent_types:
            if agent_type in self.agents:
                agent = self.agents[agent_type]
                tasks.append(agent.run(query, context))
        
        # Run all agents in parallel
        start_time = datetime.utcnow()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Aggregate results
        aggregated = {
            "query": query,
            "mode": mode,
            "agents_used": len(agent_types),
            "duration_seconds": duration,
            "results": {},
            "total_cost": 0.0,
            "total_tokens": 0,
            "errors": []
        }
        
        for i, result in enumerate(results):
            agent_type = agent_types[i]
            
            if isinstance(result, Exception):
                aggregated["errors"].append({
                    "agent": agent_type.value,
                    "error": str(result)
                })
            elif isinstance(result, AgentResult):
                aggregated["results"][agent_type.value] = result.data
                aggregated["total_cost"] += result.cost
                aggregated["total_tokens"] += result.tokens_used
                
                if not result.success:
                    aggregated["errors"].append({
                        "agent": agent_type.value,
                        "error": result.error
                    })
        
        return aggregated
    
    def get_agent_statuses(self) -> Dict[str, str]:
        """Get the current status of all agents."""
        return {
            agent_type.value: agent.status.value
            for agent_type, agent in self.agents.items()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

async def demo():
    """Demo the knowledge swarm."""
    print("=" * 60)
    print("LEGAL KNOWLEDGE SWARM DEMO")
    print("=" * 60)
    
    from configs.config import create_config
    config = create_config()
    
    # Create swarm (without LLM for demo)
    swarm = LegalKnowledgeSwarm(config)
    
    print("\n📋 Available Agents:")
    for agent_type, agent in swarm.agents.items():
        print(f"  • {agent_type.value}: {agent.description}")
    
    print("\n🔬 Research Modes:")
    for mode in ["quick", "standard", "comprehensive"]:
        agents = swarm.get_agents_for_mode(mode)
        print(f"  {mode}: {len(agents)} agents")
    
    print("\n🚀 Running Quick Research...")
    result = await swarm.research(
        query="What are the elements of breach of contract in California?",
        context={
            "state": USState.CALIFORNIA,
            "domain": LegalDomain.CONTRACT
        },
        mode="quick"
    )
    
    print(f"\n📊 Results:")
    print(f"  Mode: {result['mode']}")
    print(f"  Agents Used: {result['agents_used']}")
    print(f"  Duration: {result['duration_seconds']:.2f}s")
    print(f"  Total Cost: ${result['total_cost']:.6f}")
    print(f"  Errors: {len(result['errors'])}")


if __name__ == "__main__":
    asyncio.run(demo())
