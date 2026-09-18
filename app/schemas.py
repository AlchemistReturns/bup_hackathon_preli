from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FiniteModel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]


# ---------- Request ----------

class HourEntry(FiniteModel):
    hour: int = Field(ge=0, le=23, strict=True)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)


class BatteryConfig(FiniteModel):
    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)

    @model_validator(mode="after")
    def check_bounds(self):
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if not (self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh):
            raise ValueError("initial_energy_kwh must be within [minimum_energy_kwh, capacity_kwh]")
        return self


class ScenarioRequest(FiniteModel):
    scenario_id: str
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[HourEntry] = Field(min_length=24, max_length=24)
    battery: BatteryConfig

    @field_validator("operator_notes")
    @classmethod
    def notes_non_empty(cls, v: List[str]) -> List[str]:
        for note in v:
            if not note.strip():
                raise ValueError("operator_notes entries must be non-empty")
        return v

    @field_validator("hours")
    @classmethod
    def hours_cover_0_23(cls, v: List[HourEntry]) -> List[HourEntry]:
        seen = sorted(h.hour for h in v)
        if seen != list(range(24)):
            raise ValueError("hours must contain exactly one entry for each hour 0-23")
        return v


# ---------- Response ----------

class DirectiveInterpretation(FiniteModel):
    note_index: int = Field(ge=0, strict=True)
    applies: bool = Field(strict=True)
    directive_type: DirectiveType
    structured_adjustment: Optional[Dict[str, Any]] = None
    explanation: str = ""

    @model_validator(mode="after")
    def applies_semantics(self):
        if self.directive_type == "no_op":
            if self.applies is not False or self.structured_adjustment is not None:
                raise ValueError("no_op requires applies=false and structured_adjustment=null")
        else:
            if self.applies is not True:
                raise ValueError("non no_op directives require applies=true")
            if self.structured_adjustment is None:
                raise ValueError("non no_op directives require a structured_adjustment")
        return self


class HourlyPlanEntry(FiniteModel):
    hour: int = Field(ge=0, le=23, strict=True)
    grid_kwh: float = Field(ge=0)
    solar_used_kwh: float = Field(ge=0)
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float = Field(ge=0)
    battery_energy_after_kwh: float = Field(ge=0)

    @model_validator(mode="after")
    def idle_zero(self):
        if self.battery_action == "idle" and self.battery_kwh != 0:
            raise ValueError("battery_kwh must be 0 when battery_action is idle")
        return self


class OptimizeEnergyResponse(FiniteModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
