"""
Clinical Ontology Definition for DeepRAG Graph-based Reasoning
Defines entities, relationships, and knowledge structures for MIMIC-III clinical data
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class EntityType(Enum):
    """Enumeration of clinical entity types"""
    PATIENT = "Patient"
    CONDITION = "Condition"
    RISK_FACTOR = "RiskFactor"
    TREATMENT = "Treatment"
    ASSESSMENT = "Assessment"
    MEDICATION = "Medication"
    PROCEDURE = "Procedure"
    OUTCOME = "Outcome"


class RelationType(Enum):
    """Enumeration of relationship types between entities"""
    HAS_CONDITION = "has_condition"
    HAS_RISK_FACTOR = "has_risk_factor"
    RECEIVES_TREATMENT = "receives_treatment"
    UNDERGOES_ASSESSMENT = "undergoes_assessment"
    TAKES_MEDICATION = "takes_medication"
    UNDERGOES_PROCEDURE = "undergoes_procedure"
    HAS_OUTCOME = "has_outcome"
    CAUSED_BY = "caused_by"
    TREATED_WITH = "treated_with"
    PREVENTED_BY = "prevented_by"
    ASSOCIATED_WITH = "associated_with"
    PRECEDES = "precedes"
    FOLLOWS = "follows"


@dataclass
class EntityAttribute:
    """Represents an attribute of a clinical entity"""
    name: str
    type: str
    description: str
    required: bool = False
    enum_values: Optional[List[str]] = None


@dataclass
class EntityDefinition:
    """Defines a clinical entity type with its attributes and relationships"""
    name: str
    type: EntityType
    description: str
    attributes: List[EntityAttribute]
    can_relate_to: Dict[RelationType, List[EntityType]]


class ClinicalOntology:
    """Clinical ontology for hospital-acquired conditions"""
    
    def __init__(self):
        self.entities = self._define_entities()
        self.relationships = self._define_relationships()
        self.clinical_codes = self._define_clinical_codes()
        self.assessment_tools = self._define_assessment_tools()
    
    def _define_entities(self) -> Dict[EntityType, EntityDefinition]:
        """Define all clinical entity types"""
        
        entities = {}
        
        # Patient Entity
        entities[EntityType.PATIENT] = EntityDefinition(
            name="Patient",
            type=EntityType.PATIENT,
            description="Clinical patient entity representing a hospitalized individual",
            attributes=[
                EntityAttribute("patient_id", "integer", "Unique patient identifier (SUBJECT_ID)", required=True),
                EntityAttribute("admission_id", "integer", "Hospital admission ID (HADM_ID)"),
                EntityAttribute("age", "integer", "Patient age at admission"),
                EntityAttribute("gender", "string", "Patient gender", enum_values=["M", "F"]),
                EntityAttribute("ethnicity", "string", "Patient ethnicity"),
                EntityAttribute("admission_type", "string", "Type of hospital admission"),
                EntityAttribute("admission_location", "string", "Where patient was admitted from"),
                EntityAttribute("discharge_location", "string", "Where patient was discharged to"),
                EntityAttribute("length_of_stay", "float", "Length of hospital stay in days"),
                EntityAttribute("insurance", "string", "Insurance type")
            ],
            can_relate_to={
                RelationType.HAS_CONDITION: [EntityType.CONDITION],
                RelationType.HAS_RISK_FACTOR: [EntityType.RISK_FACTOR],
                RelationType.RECEIVES_TREATMENT: [EntityType.TREATMENT],
                RelationType.UNDERGOES_ASSESSMENT: [EntityType.ASSESSMENT],
                RelationType.TAKES_MEDICATION: [EntityType.MEDICATION],
                RelationType.UNDERGOES_PROCEDURE: [EntityType.PROCEDURE],
                RelationType.HAS_OUTCOME: [EntityType.OUTCOME]
            }
        )
        
        # Condition Entity
        entities[EntityType.CONDITION] = EntityDefinition(
            name="Condition",
            type=EntityType.CONDITION,
            description="Hospital-acquired medical condition",
            attributes=[
                EntityAttribute("code", "string", "Clinical observation code (e.g., C0392747)", required=True),
                EntityAttribute("name", "string", "Condition name", required=True),
                EntityAttribute("category", "string", "Condition category", 
                              enum_values=["HAPI", "HAAKI", "HAA"]),
                EntityAttribute("severity", "string", "Condition severity level"),
                EntityAttribute("onset_time", "float", "Time since admission to condition onset (days)"),
                EntityAttribute("icd10_code", "string", "ICD-10 diagnosis code"),
                EntityAttribute("description", "string", "Detailed condition description"),
                EntityAttribute("stage", "string", "Condition stage (for conditions like pressure injuries)"),
                EntityAttribute("location", "string", "Anatomical location of condition")
            ],
            can_relate_to={
                RelationType.CAUSED_BY: [EntityType.RISK_FACTOR],
                RelationType.TREATED_WITH: [EntityType.TREATMENT, EntityType.MEDICATION],
                RelationType.PREVENTED_BY: [EntityType.TREATMENT, EntityType.ASSESSMENT],
                RelationType.ASSOCIATED_WITH: [EntityType.PROCEDURE, EntityType.ASSESSMENT]
            }
        )
        
        # Risk Factor Entity
        entities[EntityType.RISK_FACTOR] = EntityDefinition(
            name="RiskFactor",
            type=EntityType.RISK_FACTOR,
            description="Risk factors that contribute to hospital-acquired conditions",
            attributes=[
                EntityAttribute("type", "string", "Risk factor type", required=True),
                EntityAttribute("category", "string", "Risk factor category"),
                EntityAttribute("score", "float", "Risk assessment score"),
                EntityAttribute("assessment_tool", "string", "Tool used for assessment"),
                EntityAttribute("timestamp", "datetime", "When risk factor was assessed"),
                EntityAttribute("severity", "string", "Risk severity level"),
                EntityAttribute("modifiable", "boolean", "Whether risk factor can be modified"),
                EntityAttribute("description", "string", "Detailed description of risk factor")
            ],
            can_relate_to={
                RelationType.ASSOCIATED_WITH: [EntityType.CONDITION, EntityType.ASSESSMENT]
            }
        )
        
        # Treatment Entity
        entities[EntityType.TREATMENT] = EntityDefinition(
            name="Treatment",
            type=EntityType.TREATMENT,
            description="Treatment interventions for conditions",
            attributes=[
                EntityAttribute("type", "string", "Treatment type", required=True),
                EntityAttribute("frequency", "string", "Treatment frequency"),
                EntityAttribute("duration", "float", "Treatment duration in days"),
                EntityAttribute("effectiveness", "string", "Treatment effectiveness"),
                EntityAttribute("start_date", "datetime", "Treatment start date"),
                EntityAttribute("end_date", "datetime", "Treatment end date"),
                EntityAttribute("dosage", "string", "Treatment dosage or intensity"),
                EntityAttribute("route", "string", "Route of administration"),
                EntityAttribute("provider", "string", "Healthcare provider administering treatment")
            ],
            can_relate_to={
                RelationType.ASSOCIATED_WITH: [EntityType.CONDITION, EntityType.MEDICATION, EntityType.OUTCOME]
            }
        )
        
        # Assessment Entity
        entities[EntityType.ASSESSMENT] = EntityDefinition(
            name="Assessment",
            type=EntityType.ASSESSMENT,
            description="Clinical assessments and evaluation tools",
            attributes=[
                EntityAttribute("tool_name", "string", "Assessment tool name", required=True),
                EntityAttribute("score", "float", "Assessment score"),
                EntityAttribute("interpretation", "string", "Score interpretation"),
                EntityAttribute("timestamp", "datetime", "When assessment was performed"),
                EntityAttribute("assessor", "string", "Who performed the assessment"),
                EntityAttribute("notes", "string", "Additional assessment notes"),
                EntityAttribute("risk_level", "string", "Risk level determined by assessment")
            ],
            can_relate_to={
                RelationType.ASSOCIATED_WITH: [EntityType.CONDITION, EntityType.RISK_FACTOR, EntityType.PATIENT]
            }
        )
        
        # Medication Entity
        entities[EntityType.MEDICATION] = EntityDefinition(
            name="Medication",
            type=EntityType.MEDICATION,
            description="Medications administered to patients",
            attributes=[
                EntityAttribute("name", "string", "Medication name", required=True),
                EntityAttribute("dosage", "string", "Medication dosage"),
                EntityAttribute("route", "string", "Route of administration"),
                EntityAttribute("frequency", "string", "Administration frequency"),
                EntityAttribute("start_time", "datetime", "Start time of medication"),
                EntityAttribute("end_time", "datetime", "End time of medication"),
                EntityAttribute("indication", "string", "Reason for medication"),
                EntityAttribute("nephrotoxic", "boolean", "Whether medication is nephrotoxic"),
                EntityAttribute("category", "string", "Medication category")
            ],
            can_relate_to={
                RelationType.ASSOCIATED_WITH: [EntityType.CONDITION, EntityType.TREATMENT, EntityType.RISK_FACTOR]
            }
        )
        
        return entities
    
    def _define_relationships(self) -> Dict[RelationType, Dict[str, Any]]:
        """Define relationship types with their properties"""
        
        relationships = {}
        
        for rel_type in RelationType:
            relationships[rel_type] = {
                "name": rel_type.value,
                "bidirectional": False,
                "strength": 1.0,
                "temporal": False
            }
        
        # Set specific properties for certain relationships
        relationships[RelationType.ASSOCIATED_WITH]["bidirectional"] = True
        relationships[RelationType.PRECEDES]["temporal"] = True
        relationships[RelationType.FOLLOWS]["temporal"] = True
        
        return relationships
    
    def _define_clinical_codes(self) -> Dict[str, Dict[str, Any]]:
        """Define clinical observation codes and their mappings"""
        
        return {
            "C0392747": {
                "name": "Pressure Injury/Ulcer",
                "description": "Localized damage to skin and underlying tissue over bony prominence",
                "assessment_tools": ["Braden Scale", "Norton Scale", "Waterlow Score"],
                "icd10": "L89",
                "category": "HAPI",
                "entity_type": EntityType.CONDITION
            },
            "C0022116": {
                "name": "Acute Kidney Injury",
                "description": "Sudden decrease in kidney function",
                "assessment_tools": ["KDIGO Criteria", "RIFLE Criteria", "AKIN Classification"],
                "icd10": "N17",
                "category": "HAAKI",
                "entity_type": EntityType.CONDITION
            },
            "C0002871": {
                "name": "Anemia",
                "description": "Decreased red blood cells or hemoglobin",
                "assessment_tools": ["WHO Anemia Classification", "CBC Analysis"],
                "icd10": "D64.9",
                "category": "HAA",
                "entity_type": EntityType.CONDITION
            }
        }
    
    def _define_assessment_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define clinical assessment tools and their properties"""
        
        return {
            "Braden Scale": {
                "type": "risk_assessment",
                "condition": "HAPI",
                "range": (6, 23),
                "interpretation": {
                    "high_risk": "≤ 12",
                    "moderate_risk": "13-14",
                    "mild_risk": "15-18",
                    "low_risk": "19-23"
                },
                "factors": [
                    "Sensory perception",
                    "Moisture",
                    "Activity",
                    "Mobility",
                    "Nutrition",
                    "Friction and shear"
                ]
            },
            "KDIGO Criteria": {
                "type": "diagnostic_criteria",
                "condition": "HAAKI",
                "stages": ["Stage 1", "Stage 2", "Stage 3"],
                "criteria": {
                    "serum_creatinine": "≥ 0.3 mg/dl increase within 48 hours",
                    "urine_output": "< 0.5 ml/kg/h for 6 hours"
                }
            },
            "WHO Anemia Classification": {
                "type": "diagnostic_criteria",
                "condition": "HAA",
                "thresholds": {
                    "adult_male": "< 13.0 g/dl",
                    "adult_female_non_pregnant": "< 12.0 g/dl",
                    "adult_female_pregnant": "< 11.0 g/dl"
                }
            }
        }
    
    def get_entity_definition(self, entity_type: EntityType) -> Optional[EntityDefinition]:
        """Get definition for a specific entity type"""
        return self.entities.get(entity_type)
    
    def get_related_entities(self, entity_type: EntityType, relation_type: RelationType) -> List[EntityType]:
        """Get entities that can be related to given entity via specific relationship"""
        entity_def = self.entities.get(entity_type)
        if entity_def and relation_type in entity_def.can_relate_to:
            return entity_def.can_relate_to[relation_type]
        return []
    
    def get_clinical_code_info(self, code: str) -> Optional[Dict[str, Any]]:
        """Get information about a clinical code"""
        return self.clinical_codes.get(code)
    
    def get_assessment_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about an assessment tool"""
        return self.assessment_tools.get(tool_name)
    
    def generate_graph_query_template(self, entity_type: EntityType, relation_type: RelationType) -> str:
        """Generate a Gremlin query template for graph traversal"""
        
        # Basic Gremlin query structure
        if relation_type == RelationType.HAS_CONDITION:
            return f"g.V().hasLabel('{entity_type.value}').out('{relation_type.value}').hasLabel('Condition')"
        elif relation_type == RelationType.CAUSED_BY:
            return f"g.V().hasLabel('{entity_type.value}').in('{relation_type.value}').hasLabel('RiskFactor')"
        else:
            return f"g.V().hasLabel('{entity_type.value}').both('{relation_type.value}')"
    
    def export_to_json(self) -> str:
        """Export ontology to JSON format"""
        
        ontology_dict = {
            "entities": {
                entity_type.value: {
                    "name": entity_def.name,
                    "description": entity_def.description,
                    "attributes": [
                        {
                            "name": attr.name,
                            "type": attr.type,
                            "description": attr.description,
                            "required": attr.required,
                            "enum_values": attr.enum_values
                        } for attr in entity_def.attributes
                    ],
                    "relationships": {
                        rel_type.value: [et.value for et in entity_types]
                        for rel_type, entity_types in entity_def.can_relate_to.items()
                    }
                } for entity_type, entity_def in self.entities.items()
            },
            "relationships": {
                rel_type.value: rel_props
                for rel_type, rel_props in self.relationships.items()
            },
            "clinical_codes": self.clinical_codes,
            "assessment_tools": self.assessment_tools
        }
        
        return json.dumps(ontology_dict, indent=2, default=str)
    
    def validate_entity(self, entity_type: EntityType, entity_data: Dict[str, Any]) -> List[str]:
        """Validate entity data against ontology definition"""
        
        errors = []
        entity_def = self.entities.get(entity_type)
        
        if not entity_def:
            errors.append(f"Unknown entity type: {entity_type}")
            return errors
        
        # Check required attributes
        for attr in entity_def.attributes:
            if attr.required and attr.name not in entity_data:
                errors.append(f"Required attribute missing: {attr.name}")
            
            # Check enum values
            if (attr.name in entity_data and 
                attr.enum_values and 
                entity_data[attr.name] not in attr.enum_values):
                errors.append(f"Invalid value for {attr.name}: {entity_data[attr.name]}")
        
        return errors


# Global ontology instance
clinical_ontology = ClinicalOntology()