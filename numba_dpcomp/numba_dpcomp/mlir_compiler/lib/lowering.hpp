// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir-extensions/compiler/pipeline_registry.hpp>
#include <memory>

namespace pybind11 {
    class bytes;
    class capsule;
    class object;
    class str;
    class dict;
} // namespace pybind11

struct ModuleSettings {
  bool enableGpuPipeline = false;
};

struct Module {
    mlir::MLIRContext context;
    plier::PipelineRegistry registry;
    mlir::ModuleOp module;

    Module(const ModuleSettings &settings);
};
#if defined _WIN32 || defined __CYGWIN__
  #ifdef BUILDING_DLL
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllexport))
    #else
      #define DLL_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllimport))
    #else
      #define DLL_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
    #define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define DLL_PUBLIC
    #define DLL_LOCAL
  #endif
#endif

extern DLL_PUBLIC std::shared_ptr<Module> createModule_cpp();
extern DLL_PUBLIC void compileModule_cpp(std::shared_ptr<Module> mod);

void initCompiler(pybind11::dict settings);

pybind11::capsule createModule(pybind11::dict settings);

pybind11::capsule lowerFunction(const pybind11::object &compilationContext,
                                const pybind11::capsule &pyMod,
                                const pybind11::object &funcIr);

pybind11::bytes compileModule(const pybind11::object &compilationContext,
                              const pybind11::capsule &pyMod);

pybind11::str moduleStr(const pybind11::capsule &pyMod);
