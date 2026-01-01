"""
Labor Market System

Handles employment, wages, hiring, and firing.
Based on Austrian Economics:
- Wages determined by marginal productivity
- Unemployment is caused by government intervention (minimum wage, regulations)
- Labor is a market like any other
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Set
import random
from collections import defaultdict


@dataclass
class JobPosting:
    """A job posting from a company"""
    company_id: str
    wage: Decimal
    required_skill: float
    positions: int
    sector: str


@dataclass
class JobApplication:
    """A job application from a person"""
    person_id: str
    skill_level: float
    reservation_wage: Decimal
    current_employer: Optional[str] = None


@dataclass
class EmploymentRecord:
    """Record of employment relationship"""
    person_id: str
    company_id: str
    wage: Decimal
    start_tick: int
    end_tick: Optional[int] = None


class LaborMarket:
    """
    Labor market that matches workers with jobs.

    Austrian Economics:
    - Wages are prices for labor services
    - Market clears when supply = demand at market wage
    - Government intervention (min wage) causes unemployment
    - Unions create artificial scarcity
    """

    def __init__(self, country: str = "USA"):
        self.country = country

        # Current state
        self.job_postings: List[JobPosting] = []
        self.job_applications: List[JobApplication] = []

        # Employment records
        self.employment: Dict[str, str] = {}  # person_id -> company_id
        self.company_employees: Dict[str, Set[str]] = defaultdict(set)  # company_id -> set of person_ids
        self.wages: Dict[str, Decimal] = {}  # person_id -> wage

        # Market statistics
        self.market_wage: Decimal = Decimal("1000")  # Weekly wage
        self.unemployment_rate: float = 0.0
        self.total_employed: int = 0
        self.total_labor_force: int = 0

        # History
        self.wage_history: List[Decimal] = []
        self.unemployment_history: List[float] = []

        # Government distortions
        self.minimum_wage: Decimal = Decimal("0")  # 0 = no minimum wage (free market)
        self.payroll_tax_rate: float = 0.0
        self.labor_regulations: float = 0.0  # 0-1, cost of compliance

    def set_minimum_wage(self, wage: Decimal):
        """
        Set minimum wage (government intervention).

        Austrian View: Minimum wage causes unemployment for low-skill workers.
        Any wage floor above market wage creates surplus (unemployment).
        """
        self.minimum_wage = max(Decimal("0"), wage)

    def set_payroll_tax(self, rate: float):
        """Set payroll tax rate (government extraction)"""
        self.payroll_tax_rate = max(0, min(1, rate))

    def set_labor_regulations(self, level: float):
        """Set labor regulation level (0=free, 1=heavily regulated)"""
        self.labor_regulations = max(0, min(1, level))

    # ===========================================
    # JOB POSTING AND APPLICATION
    # ===========================================

    def post_job(
        self,
        company_id: str,
        wage: Decimal,
        required_skill: float,
        positions: int,
        sector: str
    ):
        """Company posts a job opening"""
        # Apply minimum wage if set
        effective_wage = max(wage, self.minimum_wage)

        posting = JobPosting(
            company_id=company_id,
            wage=effective_wage,
            required_skill=required_skill,
            positions=positions,
            sector=sector,
        )
        self.job_postings.append(posting)

    def apply_for_job(
        self,
        person_id: str,
        skill_level: float,
        reservation_wage: Decimal,
        current_employer: Optional[str] = None
    ):
        """Person applies for jobs"""
        application = JobApplication(
            person_id=person_id,
            skill_level=skill_level,
            reservation_wage=reservation_wage,
            current_employer=current_employer,
        )
        self.job_applications.append(application)

    # ===========================================
    # MARKET CLEARING
    # ===========================================

    def clear_market(self, tick: int) -> Dict[str, any]:
        """
        Clear the labor market - match workers with jobs.

        Returns statistics about the matching.
        """
        matches = []
        unmatched_jobs = []
        unmatched_workers = []

        # Sort job postings by wage (descending) - best jobs first
        sorted_postings = sorted(
            self.job_postings,
            key=lambda p: p.wage,
            reverse=True
        )

        # Sort applications by skill (descending) - best workers first
        sorted_applications = sorted(
            self.job_applications,
            key=lambda a: a.skill_level,
            reverse=True
        )

        # Track which applications have been matched
        matched_persons: Set[str] = set()

        # Match workers to jobs
        for posting in sorted_postings:
            positions_filled = 0

            for app in sorted_applications:
                if positions_filled >= posting.positions:
                    break

                if app.person_id in matched_persons:
                    continue

                # Check if worker qualifies and accepts wage
                if (app.skill_level >= posting.required_skill and
                    posting.wage >= app.reservation_wage):

                    # Calculate effective wage after taxes and regulation costs
                    effective_wage = self._calculate_effective_wage(posting.wage)

                    # Match!
                    matches.append({
                        "person_id": app.person_id,
                        "company_id": posting.company_id,
                        "wage": effective_wage,
                        "gross_wage": posting.wage,
                    })
                    matched_persons.add(app.person_id)
                    positions_filled += 1

            if positions_filled < posting.positions:
                unmatched_jobs.append({
                    "company_id": posting.company_id,
                    "unfilled": posting.positions - positions_filled,
                })

        # Find unmatched workers
        for app in sorted_applications:
            if app.person_id not in matched_persons:
                unmatched_workers.append(app.person_id)

        # Update employment records
        for match in matches:
            self._hire(
                match["person_id"],
                match["company_id"],
                match["wage"],
                tick
            )

        # Update market wage (average of successful matches)
        if matches:
            avg_wage = sum(m["wage"] for m in matches) / len(matches)
            self.market_wage = Decimal(str(round(float(avg_wage), 2)))

        # Clear postings and applications for next tick
        self.job_postings = []
        self.job_applications = []

        # Update statistics
        self._update_statistics()

        return {
            "matches": len(matches),
            "unmatched_jobs": sum(j["unfilled"] for j in unmatched_jobs),
            "unmatched_workers": len(unmatched_workers),
            "market_wage": self.market_wage,
            "unemployment_rate": self.unemployment_rate,
        }

    def _calculate_effective_wage(self, gross_wage: Decimal) -> Decimal:
        """
        Calculate effective wage after government extractions.

        Payroll tax and regulation costs reduce what worker receives.
        """
        # Payroll tax reduces wage
        after_tax = gross_wage * Decimal(str(1 - self.payroll_tax_rate))

        # Regulation costs (compliance burden falls on wages)
        after_regulation = after_tax * Decimal(str(1 - self.labor_regulations * 0.1))

        return after_regulation

    def _hire(
        self,
        person_id: str,
        company_id: str,
        wage: Decimal,
        tick: int
    ):
        """Record a new hire"""
        # End previous employment if exists
        if person_id in self.employment:
            old_company = self.employment[person_id]
            self.company_employees[old_company].discard(person_id)

        # Create new employment
        self.employment[person_id] = company_id
        self.company_employees[company_id].add(person_id)
        self.wages[person_id] = wage

    def fire(self, person_id: str, tick: int):
        """Fire a worker"""
        if person_id in self.employment:
            company_id = self.employment[person_id]
            del self.employment[person_id]
            self.company_employees[company_id].discard(person_id)
            if person_id in self.wages:
                del self.wages[person_id]

    def quit(self, person_id: str, tick: int):
        """Worker quits"""
        self.fire(person_id, tick)

    # ===========================================
    # QUERIES
    # ===========================================

    def is_employed(self, person_id: str) -> bool:
        """Check if person is employed"""
        return person_id in self.employment

    def get_employer(self, person_id: str) -> Optional[str]:
        """Get employer of a person"""
        return self.employment.get(person_id)

    def get_wage(self, person_id: str) -> Decimal:
        """Get wage of a person"""
        return self.wages.get(person_id, Decimal("0"))

    def get_employees(self, company_id: str) -> Set[str]:
        """Get all employees of a company"""
        return self.company_employees.get(company_id, set())

    def get_employee_count(self, company_id: str) -> int:
        """Get number of employees"""
        return len(self.company_employees.get(company_id, set()))

    # ===========================================
    # INITIALIZATION
    # ===========================================

    def initialize_employment(
        self,
        persons: List,  # List of Person agents
        companies: List,  # List of Company agents
        initial_employment_rate: float = 0.85
    ) -> Dict[str, any]:
        """
        Initialize employment at start of simulation.

        Distributes workers among companies based on:
        - Company size/capital
        - Worker skills
        - Random matching (simulating historical path)
        """
        # Calculate total jobs available (based on company capital)
        total_jobs = 0
        company_jobs: Dict[str, int] = {}

        for company in companies:
            # Jobs proportional to capital stock
            jobs = max(1, int(float(company.capital_stock) / 10000))
            company_jobs[company.id] = jobs
            total_jobs += jobs

        # Target employment
        target_employed = int(len(persons) * initial_employment_rate)
        target_employed = min(target_employed, total_jobs)

        # Sort persons by skill (best get jobs first in free market)
        sorted_persons = sorted(
            persons,
            key=lambda p: p.skill_level,
            reverse=True
        )

        # Assign workers to companies
        employed_count = 0
        company_idx = 0
        company_list = list(companies)

        for person in sorted_persons:
            if employed_count >= target_employed:
                break

            # Find a company with open positions
            attempts = 0
            while attempts < len(companies):
                company = company_list[company_idx % len(company_list)]
                current_employees = self.get_employee_count(company.id)

                if current_employees < company_jobs.get(company.id, 0):
                    # Hire!
                    base_wage = Decimal("800") + Decimal(str(person.skill_level * 400))
                    wage = max(base_wage, self.minimum_wage)

                    self._hire(person.id, company.id, wage, tick=0)

                    # Update person's state
                    person.employer_id = company.id
                    person.wage = wage
                    person.state.employed = True

                    # Update company's employee list
                    if person.id not in company.employees:
                        company.employees.append(person.id)

                    employed_count += 1
                    break

                company_idx += 1
                attempts += 1

            company_idx += 1

        # Update statistics
        self.total_labor_force = len(persons)
        self.total_employed = employed_count
        self._update_statistics()

        return {
            "total_persons": len(persons),
            "total_companies": len(companies),
            "total_jobs_available": total_jobs,
            "employed": employed_count,
            "unemployed": len(persons) - employed_count,
            "employment_rate": employed_count / len(persons) if persons else 0,
            "unemployment_rate": self.unemployment_rate,
        }

    def _update_statistics(self):
        """Update market statistics"""
        self.total_employed = len(self.employment)

        if self.total_labor_force > 0:
            self.unemployment_rate = 1 - (self.total_employed / self.total_labor_force)
        else:
            self.unemployment_rate = 0.0

        # Record history
        self.wage_history.append(self.market_wage)
        self.unemployment_history.append(self.unemployment_rate)

        # Keep history manageable
        if len(self.wage_history) > 1000:
            self.wage_history.pop(0)
        if len(self.unemployment_history) > 1000:
            self.unemployment_history.pop(0)

    def get_statistics(self) -> Dict[str, any]:
        """Get current labor market statistics"""
        return {
            "market_wage": str(self.market_wage),
            "unemployment_rate": self.unemployment_rate,
            "total_employed": self.total_employed,
            "total_labor_force": self.total_labor_force,
            "minimum_wage": str(self.minimum_wage),
            "payroll_tax_rate": self.payroll_tax_rate,
            "labor_regulations": self.labor_regulations,
            "wage_trend": self._calculate_wage_trend(),
        }

    def _calculate_wage_trend(self) -> str:
        """Calculate wage trend"""
        if len(self.wage_history) < 5:
            return "stable"

        recent = sum(self.wage_history[-5:]) / 5
        older = sum(self.wage_history[-10:-5]) / 5 if len(self.wage_history) >= 10 else recent

        if recent > older * Decimal("1.02"):
            return "rising"
        elif recent < older * Decimal("0.98"):
            return "falling"
        return "stable"
