#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace {

using llvm::StringRef;
using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

constexpr StringRef ArtifactNames[] = {"contract", "target", "candidate",
                                       "estimate", "measurement", "trace"};

bool fail(const llvm::Twine &Message) {
  llvm::errs() << "ocf-contract-check: " << Message << '\n';
  return false;
}

bool appendCodePoint(uint32_t CodePoint, std::string &Output) {
  auto AppendHex16 = [&](uint16_t Value) {
    static constexpr char Hex[] = "0123456789abcdef";
    Output += "\\u";
    for (int Shift = 12; Shift >= 0; Shift -= 4)
      Output.push_back(Hex[(Value >> Shift) & 0xF]);
  };
  if (CodePoint <= 0xFFFF) {
    AppendHex16(static_cast<uint16_t>(CodePoint));
    return true;
  }
  if (CodePoint > 0x10FFFF)
    return false;
  CodePoint -= 0x10000;
  AppendHex16(static_cast<uint16_t>(0xD800 + (CodePoint >> 10)));
  AppendHex16(static_cast<uint16_t>(0xDC00 + (CodePoint & 0x3FF)));
  return true;
}

bool appendCanonicalString(StringRef Input, std::string &Output) {
  Output.push_back('"');
  for (size_t Index = 0; Index < Input.size();) {
    const uint8_t First = static_cast<uint8_t>(Input[Index++]);
    if (First < 0x80) {
      switch (First) {
      case '"':
        Output += "\\\"";
        break;
      case '\\':
        Output += "\\\\";
        break;
      case '\b':
        Output += "\\b";
        break;
      case '\f':
        Output += "\\f";
        break;
      case '\n':
        Output += "\\n";
        break;
      case '\r':
        Output += "\\r";
        break;
      case '\t':
        Output += "\\t";
        break;
      default:
        if (First < 0x20) {
          if (!appendCodePoint(First, Output))
            return false;
        } else {
          Output.push_back(static_cast<char>(First));
        }
      }
      continue;
    }

    unsigned Continuations = 0;
    uint32_t CodePoint = 0;
    if ((First & 0xE0) == 0xC0) {
      Continuations = 1;
      CodePoint = First & 0x1F;
    } else if ((First & 0xF0) == 0xE0) {
      Continuations = 2;
      CodePoint = First & 0x0F;
    } else if ((First & 0xF8) == 0xF0) {
      Continuations = 3;
      CodePoint = First & 0x07;
    } else {
      return false;
    }
    if (Index + Continuations > Input.size())
      return false;
    for (unsigned I = 0; I < Continuations; ++I) {
      const uint8_t Byte = static_cast<uint8_t>(Input[Index++]);
      if ((Byte & 0xC0) != 0x80)
        return false;
      CodePoint = (CodePoint << 6) | (Byte & 0x3F);
    }
    if (!appendCodePoint(CodePoint, Output))
      return false;
  }
  Output.push_back('"');
  return true;
}

std::string normalizeExponent(std::string Number) {
  const size_t Exponent = Number.find_first_of("eE");
  if (Exponent == std::string::npos)
    return Number;
  Number[Exponent] = 'e';
  size_t Digits = Exponent + 1;
  if (Digits < Number.size() &&
      (Number[Digits] == '+' || Number[Digits] == '-'))
    ++Digits;
  while (Number.size() - Digits < 2)
    Number.insert(Digits, 1, '0');
  return Number;
}

bool appendCanonicalNumber(const Value &Input, std::string &Output) {
  std::string Printed;
  llvm::raw_string_ostream Stream(Printed);
  Input.print(Stream);
  Stream.flush();
  if (Printed == "-0") {
    Output.push_back('0');
    return true;
  }
  if (Printed.find_first_of(".eE") == std::string::npos) {
    Output += Printed;
    return true;
  }
  const std::optional<double> Number = Input.getAsNumber();
  if (!Number || !std::isfinite(*Number))
    return false;
  std::array<char, 512> Buffer{};
  std::to_chars_result Result;
  if (*Number == 0.0) {
    Output.push_back('0');
    return true;
  }
  if (std::trunc(*Number) == *Number) {
    Result = std::to_chars(Buffer.data(), Buffer.data() + Buffer.size(),
                           *Number, std::chars_format::fixed, 0);
  } else {
    Result = std::to_chars(Buffer.data(), Buffer.data() + Buffer.size(),
                           *Number, std::chars_format::general);
  }
  if (Result.ec != std::errc())
    return false;
  Output += normalizeExponent(std::string(Buffer.data(), Result.ptr));
  return true;
}

bool appendCanonical(const Value &Input, std::string &Output) {
  switch (Input.kind()) {
  case Value::Null:
    Output += "null";
    return true;
  case Value::Boolean:
    Output += *Input.getAsBoolean() ? "true" : "false";
    return true;
  case Value::Number:
    return appendCanonicalNumber(Input, Output);
  case Value::String:
    return appendCanonicalString(*Input.getAsString(), Output);
  case Value::Array: {
    Output.push_back('[');
    bool First = true;
    for (const Value &Element : *Input.getAsArray()) {
      if (!First)
        Output.push_back(',');
      First = false;
      if (!appendCanonical(Element, Output))
        return false;
    }
    Output.push_back(']');
    return true;
  }
  case Value::Object: {
    Output.push_back('{');
    bool First = true;
    for (const Object::value_type *Element :
         llvm::json::sortedElements(*Input.getAsObject())) {
      if (!First)
        Output.push_back(',');
      First = false;
      if (!appendCanonicalString(Element->first, Output))
        return false;
      Output.push_back(':');
      if (!appendCanonical(Element->second, Output))
        return false;
    }
    Output.push_back('}');
    return true;
  }
  }
  return false;
}

std::optional<std::string> fingerprint(const Value &Input) {
  std::string Canonical;
  if (!appendCanonical(Input, Canonical))
    return std::nullopt;
  const auto Digest = llvm::SHA256::hash(llvm::arrayRefFromStringRef(Canonical));
  return llvm::toHex(Digest, true);
}

const Object *requireObject(const Object &Parent, StringRef Key) {
  const Object *Result = Parent.getObject(Key);
  if (!Result)
    fail(llvm::Twine("missing object field '") + Key + "'");
  return Result;
}

bool requireStringEquals(const Object &Parent, StringRef Key,
                         StringRef Expected) {
  const std::optional<StringRef> Actual = Parent.getString(Key);
  if (!Actual)
    return fail(llvm::Twine("missing string field '") + Key + "'");
  if (*Actual != Expected)
    return fail(llvm::Twine("field '") + Key + "' expected '" + Expected +
                "' but got '" + *Actual + "'");
  return true;
}

bool requireSchemaVersion(const Object &Artifact) {
  const std::optional<int64_t> Version = Artifact.getInteger("schema_version");
  return Version && *Version == OCF_SCHEMA_VERSION
             ? true
             : fail("unsupported or missing schema_version");
}

std::optional<std::vector<int64_t>> integerArray(const Object &Parent,
                                                 StringRef Key,
                                                 size_t ExpectedSize) {
  const Array *Values = Parent.getArray(Key);
  if (!Values || Values->size() != ExpectedSize) {
    fail(llvm::Twine("field '") + Key + "' has an invalid length");
    return std::nullopt;
  }
  std::vector<int64_t> Result;
  Result.reserve(Values->size());
  for (const Value &Item : *Values) {
    const std::optional<int64_t> Number = Item.getAsInteger();
    if (!Number) {
      fail(llvm::Twine("field '") + Key + "' must contain integers");
      return std::nullopt;
    }
    Result.push_back(*Number);
  }
  return Result;
}

bool validateContract(const Object &Contract) {
  if (!requireSchemaVersion(Contract) ||
      !requireStringEquals(Contract, "op", "conv2d") ||
      !requireStringEquals(Contract, "algorithm", "direct") ||
      !requireStringEquals(Contract, "input_layout", "NCHW") ||
      !requireStringEquals(Contract, "filter_layout", "OIHW") ||
      !requireStringEquals(Contract, "output_layout", "NCHW") ||
      !requireStringEquals(Contract, "input_dtype", "f32") ||
      !requireStringEquals(Contract, "filter_dtype", "f32") ||
      !requireStringEquals(Contract, "output_dtype", "f32") ||
      !requireStringEquals(Contract, "accumulation_dtype", "f32"))
    return false;

  const auto Input = integerArray(Contract, "input_shape", 4);
  const auto Filter = integerArray(Contract, "filter_shape", 4);
  const auto Output = integerArray(Contract, "output_shape", 4);
  const auto Strides = integerArray(Contract, "strides", 2);
  const auto Padding = integerArray(Contract, "padding", 4);
  const auto Dilation = integerArray(Contract, "dilation", 2);
  if (!Input || !Filter || !Output || !Strides || !Padding || !Dilation)
    return false;
  const auto Groups = Contract.getInteger("groups");
  if (!Groups || (*Input)[1] != (*Filter)[1] || *Groups != 1 ||
      (*Dilation)[0] != 1 || (*Dilation)[1] != 1)
    return fail("Conv2D channel, group, or dilation contract mismatch");
  if (std::any_of(Input->begin(), Input->end(), [](int64_t V) { return V <= 0; }) ||
      std::any_of(Filter->begin(), Filter->end(), [](int64_t V) { return V <= 0; }) ||
      std::any_of(Strides->begin(), Strides->end(), [](int64_t V) { return V <= 0; }) ||
      std::any_of(Padding->begin(), Padding->end(), [](int64_t V) { return V < 0; }))
    return fail("Conv2D shape, stride, or padding is outside the MVP contract");
  const int64_t ExpectedH =
      ((*Input)[2] + (*Padding)[0] + (*Padding)[1] - (*Filter)[2]) /
          (*Strides)[0] +
      1;
  const int64_t ExpectedW =
      ((*Input)[3] + (*Padding)[2] + (*Padding)[3] - (*Filter)[3]) /
          (*Strides)[1] +
      1;
  if ((*Output)[0] != (*Input)[0] || (*Output)[1] != (*Filter)[0] ||
      (*Output)[2] != ExpectedH || (*Output)[3] != ExpectedW || ExpectedH <= 0 ||
      ExpectedW <= 0)
    return fail("Conv2D output_shape does not match the inferred shape");
  return true;
}

bool validateTarget(const Object &Target) {
  if (!requireSchemaVersion(Target))
    return false;
  const auto TargetID = Target.getString("target_id");
  const auto Backend = Target.getString("backend");
  const auto ProfileVersion = Target.getInteger("profile_version");
  const Object *Resources = Target.getObject("resources");
  if (!TargetID || TargetID->empty() || !Backend || Backend->empty() ||
      !ProfileVersion || *ProfileVersion <= 0 || !Resources || Resources->empty())
    return fail("target identity or resources are invalid");
  return true;
}

bool validateCandidate(const Object &Candidate,
                       const std::map<std::string, std::string> &Hashes,
                       const Object &Target) {
  if (!requireSchemaVersion(Candidate) ||
      !requireStringEquals(Candidate, "compute_contract_ref", Hashes.at("contract")) ||
      !requireStringEquals(Candidate, "target_profile_fingerprint", Hashes.at("target")) ||
      !requireStringEquals(Candidate, "algorithm", "direct"))
    return false;
  const auto TargetID = Target.getString("target_id");
  const auto ProfileVersion = Target.getInteger("profile_version");
  if (!TargetID || !ProfileVersion)
    return false;
  const std::string TargetRef =
      (llvm::Twine(*TargetID) + "@" + llvm::Twine(*ProfileVersion)).str();
  if (!requireStringEquals(Candidate, "target_profile_ref", TargetRef))
    return false;
  const Object *Tiles = requireObject(Candidate, "tiles");
  const Object *Provenance = requireObject(Candidate, "decomposition_provenance");
  if (!Tiles || !Provenance)
    return false;
  for (StringRef Axis : {StringRef("oc"), StringRef("oh"), StringRef("ow")}) {
    const auto Tile = Tiles->getInteger(Axis);
    if (!Tile || *Tile <= 0)
      return fail(llvm::Twine("invalid tile for axis ") + Axis);
  }
  return requireStringEquals(*Provenance, "name", "direct") &&
         requireStringEquals(*Provenance, "source_operator_ref", Hashes.at("contract"));
}

bool validateEstimate(const Object &Estimate,
                      const std::map<std::string, std::string> &Hashes) {
  if (!requireSchemaVersion(Estimate) ||
      !requireStringEquals(Estimate, "candidate_ref", Hashes.at("candidate")))
    return false;
  const auto Confidence = Estimate.getNumber("confidence");
  const auto Latency = Estimate.getNumber("latency_cycles");
  if (!Confidence || *Confidence < 0.0 || *Confidence > 1.0 || !Latency ||
      *Latency < 0.0)
    return fail("estimate confidence or latency is invalid");
  return true;
}

bool validateMeasurement(const Object &Measurement,
                         const std::map<std::string, std::string> &Hashes) {
  if (!requireSchemaVersion(Measurement) ||
      !requireStringEquals(Measurement, "candidate_ref", Hashes.at("candidate")) ||
      !requireStringEquals(Measurement, "target_profile_fingerprint", Hashes.at("target")))
    return false;
  const auto Source = Measurement.getString("source_kind");
  if (!Source || (*Source != "real_hardware" && *Source != "cycle_accurate_model"))
    return fail("measurement source_kind is not a qualified source");
  const auto Metric = Measurement.getString("metric");
  const auto Unit = Measurement.getString("unit");
  if (!Metric || !Unit ||
      ((*Metric == "latency_cycles" && *Unit != "cycles") ||
       (*Metric == "latency_us" && *Unit != "us") ||
       (*Metric != "latency_cycles" && *Metric != "latency_us")))
    return fail("measurement metric and unit are inconsistent");
  const Array *Samples = Measurement.getArray("samples");
  if (!Samples || Samples->empty())
    return fail("measurement samples are missing");
  std::vector<double> Values;
  Values.reserve(Samples->size());
  for (const Value &Sample : *Samples) {
    const auto Number = Sample.getAsNumber();
    if (!Number || *Number < 0.0)
      return fail("measurement sample is invalid");
    Values.push_back(*Number);
  }
  std::sort(Values.begin(), Values.end());
  const Object *Summary = requireObject(Measurement, "summary");
  if (!Summary)
    return false;
  const double Median = Values.size() % 2 == 0
                            ? (Values[Values.size() / 2 - 1] +
                               Values[Values.size() / 2]) /
                                  2.0
                            : Values[Values.size() / 2];
  const size_t P90Index =
      std::max<size_t>(1, static_cast<size_t>(std::ceil(0.9 * Values.size()))) - 1;
  const auto Min = Summary->getNumber("min");
  const auto SummaryMedian = Summary->getNumber("median");
  const auto P90 = Summary->getNumber("p90");
  const auto Max = Summary->getNumber("max");
  if (!Min || !SummaryMedian || !P90 || !Max || *Min != Values.front() ||
      *SummaryMedian != Median || *P90 != Values[P90Index] || *Max != Values.back())
    return fail("measurement summary does not match samples");
  return true;
}

bool arrayContains(const Array &Values, StringRef Expected) {
  return std::any_of(Values.begin(), Values.end(), [&](const Value &Item) {
    const auto String = Item.getAsString();
    return String && *String == Expected;
  });
}

bool validateTrace(const Object &Trace,
                   const std::map<std::string, std::string> &Hashes) {
  if (!requireSchemaVersion(Trace) ||
      !requireStringEquals(Trace, "input_fingerprint", Hashes.at("contract")) ||
      !requireStringEquals(Trace, "target_profile_fingerprint", Hashes.at("target")))
    return false;
  const Array *Measurements = Trace.getArray("measurement_refs");
  const Array *Decisions = Trace.getArray("decisions");
  if (!Measurements || !arrayContains(*Measurements, Hashes.at("measurement")) ||
      !Decisions || Decisions->size() != 1)
    return fail("trace does not reference the expected measurement and decision");
  const Object *Decision = (*Decisions)[0].getAsObject();
  if (!Decision ||
      !requireStringEquals(*Decision, "selected_ref", Hashes.at("candidate")))
    return false;
  const Object *EstimateRefs = requireObject(*Decision, "estimate_refs");
  return EstimateRefs &&
         requireStringEquals(*EstimateRefs, Hashes.at("candidate"), Hashes.at("estimate"));
}

std::optional<Value> parseJSONFile(StringRef Path) {
  auto Buffer = llvm::MemoryBuffer::getFile(Path);
  if (!Buffer) {
    fail(llvm::Twine("cannot read ") + Path + ": " + Buffer.getError().message());
    return std::nullopt;
  }
  llvm::Expected<Value> Parsed = llvm::json::parse((*Buffer)->getBuffer());
  if (!Parsed) {
    llvm::errs() << "ocf-contract-check: invalid JSON in " << Path << ": ";
    llvm::logAllUnhandledErrors(Parsed.takeError(), llvm::errs());
    return std::nullopt;
  }
  return std::move(*Parsed);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    llvm::errs() << "usage: ocf-contract-check <bundle.json> <golden.json>\n";
    return 2;
  }
  std::optional<Value> BundleValue = parseJSONFile(argv[1]);
  std::optional<Value> GoldenValue = parseJSONFile(argv[2]);
  if (!BundleValue || !GoldenValue)
    return 1;
  const Object *Bundle = BundleValue->getAsObject();
  const Object *Golden = GoldenValue->getAsObject();
  if (!Bundle || !Golden) {
    fail("bundle and golden roots must be objects");
    return 1;
  }

  std::map<std::string, std::string> Hashes;
  std::map<std::string, const Object *> Artifacts;
  for (StringRef Name : ArtifactNames) {
    const Value *ArtifactValue = Bundle->get(Name);
    const Object *Artifact = ArtifactValue ? ArtifactValue->getAsObject() : nullptr;
    const std::optional<StringRef> Expected = Golden->getString(Name);
    if (!ArtifactValue || !Artifact || !Expected) {
      fail(llvm::Twine("missing artifact or golden fingerprint for ") + Name);
      return 1;
    }
    const std::optional<std::string> Actual = fingerprint(*ArtifactValue);
    if (!Actual) {
      fail(llvm::Twine("cannot canonicalize artifact ") + Name);
      return 1;
    }
    Hashes.emplace(Name.str(), *Actual);
    Artifacts.emplace(Name.str(), Artifact);
  }

  if (!validateContract(*Artifacts.at("contract")) ||
      !validateTarget(*Artifacts.at("target")) ||
      !validateCandidate(*Artifacts.at("candidate"), Hashes,
                         *Artifacts.at("target")) ||
      !validateEstimate(*Artifacts.at("estimate"), Hashes) ||
      !validateMeasurement(*Artifacts.at("measurement"), Hashes) ||
      !validateTrace(*Artifacts.at("trace"), Hashes))
    return 1;

  for (StringRef Name : ArtifactNames) {
    const StringRef Expected = *Golden->getString(Name);
    if (Hashes.at(Name.str()) != Expected) {
      fail(llvm::Twine("fingerprint mismatch for ") + Name + ": expected " +
           Expected + ", got " + Hashes.at(Name.str()));
      return 1;
    }
    llvm::outs() << Name << ' ' << Expected << '\n';
  }
  return 0;
}
