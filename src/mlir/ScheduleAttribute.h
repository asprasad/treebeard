#ifndef _SCHEDULEATTRIBUTE_H_
#define _SCHEDULEATTRIBUTE_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "schedule.h"

namespace mlir 
{
namespace decisionforest
{
namespace detail
{
struct ScheduleAttrStorage : public ::mlir::AttributeStorage
{
    ScheduleAttrStorage(::mlir::Type type, mlir::decisionforest::Schedule* schedule)
      : ::mlir::AttributeStorage(type), m_schedule(schedule) { }

    using KeyTy = std::tuple<::mlir::Type, mlir::decisionforest::Schedule*>;

    bool operator==(const KeyTy &tblgenKey) const {
        if (!(getType() == std::get<0>(tblgenKey)))
            return false;
        if (!(m_schedule == std::get<1>(tblgenKey)))
            return false;
        return true;
    }
    
    static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
        auto schedule = std::get<1>(tblgenKey);
        return ::llvm::hash_combine(std::get<0>(tblgenKey), schedule);
    }

    static ScheduleAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator,
                                          const KeyTy &tblgenKey) {
        auto type = std::get<0>(tblgenKey);
        auto schedule = std::get<1>(tblgenKey);
        return new (allocator.allocate<DecisionTreeAttrStorage>())
                    ScheduleAttrStorage(type, schedule);
    }
    mlir::decisionforest::Schedule* m_schedule;
};

} // namespace detail

class ScheduleType : public mlir::Type::TypeBase<ScheduleType, mlir::Type, TypeStorage>
{
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;
    void print(mlir::DialectAsmPrinter &printer) { printer << "ScheduleType"; }   
};

class ScheduleAttribute : public ::mlir::Attribute::AttrBase<ScheduleAttribute, ::mlir::Attribute,
                                                             detail::ScheduleAttrStorage>
{
public:
    /// Inherit some necessary constructors from 'AttrBase'.
    using Base::Base;
    // using ValueType = APInt;
    static ScheduleAttribute get(Type type,  mlir::decisionforest::Schedule* schedule) {
        return Base::get(type.getContext(), type, schedule);
    }
    const Schedule* GetSchedule() { return getImpl()->m_schedule; }
    
    void Print(::mlir::DialectAsmPrinter &os) {
        std::string scheduleStr = getImpl()->m_schedule->PrintToString();
        os << "Schedule = { " << scheduleStr << " }";
    }
};

} // decisionforest
} // mlir

#endif