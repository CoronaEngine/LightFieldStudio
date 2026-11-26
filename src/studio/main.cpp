#include "backend/frame_contract.h"
#include "base/scattering/material.h"
#include "ssat/denoise_pipeline.h"
#include "ssat/denoise_presets.h"
#include "ssat/phase1/light_field_processor.h"
#include "lightfield_framebuffer.h"
#include "ssat/resident_adapter.h"
#include "ssat/resident_policy.h"
#include "render_scene_transform.h"

#include <glad.h>
#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cuda.h>
#include <windows.h>
#include "core/util/logging.h"
#include <spdlog/sinks/basic_file_sink.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <commdlg.h>
#include "camera_controller.h"
#include "interop_presenter.h"
#include "scene_presets.h"
#include "ssat/runtime_paths.h"

namespace {
std::filesystem::path choose_scene(HWND owner) {
    wchar_t file[32768]{};OPENFILENAMEW dialog{};dialog.lStructSize=sizeof(dialog);
    dialog.hwndOwner=owner;dialog.lpstrFilter=L"Vision scene (*.json)\0*.json\0\0";
    dialog.lpstrFile=file;dialog.nMaxFile=32768;dialog.Flags=OFN_FILEMUSTEXIST|OFN_PATHMUSTEXIST|OFN_NOCHANGEDIR;
    return GetOpenFileNameW(&dialog)?std::filesystem::path(file):std::filesystem::path{};
}
} // namespace

int main(int argc,char **argv) try {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    using namespace lfs::studio;
    const auto runtime = lfs::denoise::runtime_directory();
    const auto user = lfs::denoise::user_data_directory();
    std::filesystem::create_directories(user/"logs");
    const auto log_path=user/"logs"/("studio-"+std::to_string(GetCurrentProcessId())+".log");
    auto log_sink=std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path.string(),true);
    auto &logger=ocarina::logger();logger.sinks().clear();logger.sinks().push_back(log_sink);
    logger.flush_on(spdlog::level::info);
    std::filesystem::path scene;
    auto lane=lfs::denoise::Lane::Quality;
    int resolution=0;
    int selected_scene=0;
    bool fullscreen=false;
    for(int i=1;i<argc;++i) {
        const std::string argument=argv[i];
        if (argument=="--scene" && i+1<argc) {
            const std::string value=argv[++i];
            selected_scene=bundled_index(value);
            if (selected_scene<0) scene=std::filesystem::absolute(std::filesystem::u8path(value));
        }
        else if(argument=="--resolution"&&i+1<argc) {
            std::string id=argv[++i];
            if(id.size()==2u && id.back()=='K')id.back()='k';
            const auto mode=std::find_if(display_modes.begin(),display_modes.end(),[&](const DisplayMode &m){return id==m.id;});
            if(mode==display_modes.end())throw std::runtime_error("Expected resolution 2k, 4k or 8k.");
            resolution=int(mode-display_modes.begin());
        }
        else if(argument=="--performance")lane=lfs::denoise::Lane::Performance;
        else if(argument=="--fullscreen")fullscreen=true;
        else if(argument=="--help") {std::cout<<"LightFieldStudio [--scene cbox|coffee|spaceship|file.json] [--resolution 2k|4k|8k] [--performance] [--fullscreen]\n";return 0;}
        else throw std::runtime_error("Unknown argument: "+argument);
    }
    uint32_t render_width=display_modes[resolution].width,render_height=display_modes[resolution].height;
    if (selected_scene>=0) scene=bundled_path(runtime,selected_scene,display_modes[resolution]);
    if(!std::filesystem::is_regular_file(scene))throw std::runtime_error("The selected scene is missing. Extract the complete portable ZIP or download the scene assets.");
    std::filesystem::current_path(user);
    if(cuInit(0)!=CUDA_SUCCESS)throw std::runtime_error("NVIDIA CUDA is unavailable. Install a compatible NVIDIA driver.");
    CUdevice device{};int major=0,minor=0;
    if(cuDeviceGet(&device,0)!=CUDA_SUCCESS)throw std::runtime_error("No NVIDIA GPU is available.");
    cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device);
    cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device);
    if(major*10+minor<75)throw std::runtime_error("LightFieldStudio requires an RTX 20-series or newer NVIDIA GPU.");
    if(!glfwInit())throw std::runtime_error("GLFW initialization failed.");
    glfwWindowHint(GLFW_SCALE_TO_MONITOR,GLFW_TRUE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,1);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow *controller=glfwCreateWindow(600,640,"LightFieldStudio",nullptr,nullptr);
    if(!controller)throw std::runtime_error("Cannot create the control window.");
    GLFWwindow *output=glfwCreateWindow(1280,720,"LightFieldStudio - LFD output",nullptr,controller);
    if(!output)throw std::runtime_error("Cannot create the LFD output window.");
    glfwSetWindowPos(controller,24,48);glfwSetWindowPos(output,660,48);
    glfwMakeContextCurrent(controller);
    if(!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))throw std::runtime_error("Cannot load OpenGL.");
    IMGUI_CHECKVERSION();ImGui::CreateContext();ImGui::StyleColorsDark();
    float scale_x=1,scale_y=1;glfwGetWindowContentScale(controller,&scale_x,&scale_y);
    ImGui::GetStyle().ScaleAllSizes(scale_x);ImGui::GetIO().FontGlobalScale=scale_x;
    ImGui_ImplGlfw_InitForOpenGL(controller,true);ImGui_ImplOpenGL3_Init("#version 330");
    const auto settings=(user/"imgui.ini").string();ImGui::GetIO().IniFilename=settings.c_str();
    const auto session=user/"sessions"/std::to_string(GetCurrentProcessId());
    std::filesystem::create_directories(session);
    lfs::backend::SessionConfig config{};config.runtime_directory=runtime;
    config.shader_cache_directory=user/"shader-cache";config.debug_name="LightFieldStudio";
    std::unique_ptr<lfs::backend::RendererSession> adapter;
    std::size_t physical_device_bytes{};
    if(cuDeviceTotalMem(&physical_device_bytes,device)!=CUDA_SUCCESS)throw std::runtime_error("Cannot query the renderer device capacity.");
    std::unique_ptr<lfs::denoise::DenoisePipeline> pipeline;
    std::unique_ptr<lfs::denoise::resident::ResidentAdapter> resident_processor;
    std::unique_ptr<lfs::lightfield::LightFieldProcessor> processor;
    lfs::LenticularParams calibration{};lfs::LightFieldGeometryParams geometry{};
    CameraController camera{};InteropPresenter presenter;
    std::filesystem::path active_scene;uint64_t frame=0,generation=0;
    bool rebuild=true,load=true,reload_calibration=true,reload_work_scene=false,reset=true;int monitor_index=0;
    bool compiling=true;std::string error,settings_error;double frame_ms=0,fps=0;
    auto previous=std::chrono::steady_clock::now();
    std::optional<std::chrono::steady_clock::time_point> previous_presentation;
    glfwSetWindowUserPointer(output,&camera);
    glfwGetCursorPos(output,&camera.last_mouse_x,&camera.last_mouse_y);
    glfwSetMouseButtonCallback(output,[](GLFWwindow *w,int button,int action,int){static_cast<CameraController *>(glfwGetWindowUserPointer(w))->mouse_button(w,button,action);});
    glfwSetCursorPosCallback(output,[](GLFWwindow *w,double x,double y){static_cast<CameraController *>(glfwGetWindowUserPointer(w))->cursor(x,y);});
    glfwSetScrollCallback(output,[](GLFWwindow *w,double,double y){static_cast<CameraController *>(glfwGetWindowUserPointer(w))->add_scroll(y);});
    glfwSetKeyCallback(output,[](GLFWwindow *w,int key,int,int action,int){
        if(action!=GLFW_PRESS)return;
        if(key==GLFW_KEY_F11)
            static_cast<CameraController *>(glfwGetWindowUserPointer(w))->toggle_fullscreen_requested=true;
        else if(key==GLFW_KEY_ESCAPE)glfwSetWindowShouldClose(w,GLFW_TRUE);
    });
    auto show_loading=[&](const char *message) {
        glfwMakeContextCurrent(controller);
        ImGui_ImplOpenGL3_NewFrame();ImGui_ImplGlfw_NewFrame();ImGui::NewFrame();
        int w,h;glfwGetWindowSize(controller,&w,&h);
        ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize({float(w),float(h)});
        ImGui::Begin("LightFieldStudio",nullptr,ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse);
        ImGui::TextWrapped("%s",message);
        ImGui::TextWrapped("The first launch compiles shaders. Please wait. Later launches reuse the local cache.");
        ImGui::End();ImGui::Render();glfwGetFramebufferSize(controller,&w,&h);
        glViewport(0,0,w,h);glClearColor(.035f,.04f,.05f,1);glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());glfwSwapBuffers(controller);glfwPollEvents();
    };
    while(!glfwWindowShouldClose(controller)&&!glfwWindowShouldClose(output)) {
        glfwPollEvents();
        if(camera.toggle_fullscreen_requested) {
            fullscreen=!fullscreen;
            camera.toggle_fullscreen_requested=false;
        }
        const auto now=std::chrono::steady_clock::now();
        const float dt=std::clamp(std::chrono::duration<float>(now-previous).count(),0.f,.1f);previous=now;
        if(rebuild) {
            previous_presentation.reset();
            fps=0;
            glfwSetWindowTitle(output,"LightFieldStudio - compiling shaders, please wait");
            show_loading("Loading scene and preparing shaders...");
            try {
                processor.reset();resident_processor.reset();pipeline.reset();
                glfwMakeContextCurrent(output);presenter.shutdown();
                if(load || reload_calibration) {
                    std::vector<std::string> errors;
                    calibration=lfs::load_lenticular_params_from_scene(scene,errors);
                    geometry=lfs::load_lightfield_geometry_from_scene(scene,errors);
                    if(!errors.empty())throw std::runtime_error(errors.front());
                    const auto &mode=display_modes[resolution];
                    if (calibration.width!=mode.width || calibration.height!=mode.height) {
                        calibration=lfs::load_lenticular_params_from_scene(bundled_path(runtime,0,mode),errors);
                        if (!errors.empty()) throw std::runtime_error(errors.front());
                    }
                    reload_calibration=false;
                }
                if(load) {
                    const auto c=lfs::load_scene_camera_state(scene);
                    if(!c.ok)throw std::runtime_error(c.errors.empty()?"Invalid scene camera.":c.errors.front());
                    camera.base.position=c.position;camera.base.target=c.target;camera.base.up=c.up;
                    camera.base.vertical_fov_degrees=c.vertical_fov_degrees;camera.base.override_scene_camera=true;camera.reset();
                }
                geometry.aspect=double(render_width)/double(render_height);
                geometry.focal_width=2.0*geometry.focal_distance*std::tan(geometry.fov_h_degrees*3.14159265358979323846/360.0);
                geometry.focal_height=geometry.focal_width/geometry.aspect;
                // Resolution changes can select different framebuffer storage.
                const bool resident=lfs::denoise::resident::requires_resident_framebuffer(render_width,render_height,physical_device_bytes);
                if(load || reload_work_scene || active_scene.empty()) {
                    const auto prepared=lfs::prepare_render_scene(scene,session/std::to_string(generation+1)/"scene.json",render_width,render_height,resident?lfs::StudioFramebuffer::Resident:lfs::StudioFramebuffer::Normal);
                    if(!prepared.ok)throw std::runtime_error(prepared.errors.empty()?"Cannot prepare the scene.":prepared.errors.front());
                    active_scene=prepared.output_scene;
                }
                if(!adapter) {
                    auto result=lfs::backend::create_renderer_session(config);
                    if(!result.ok())throw std::runtime_error(result.error().message);
                    adapter=std::move(result).value();
                }
                if(load || reload_work_scene) {
                    // Material lookup textures are registered with the outgoing
                    // pipeline's bindless array. Rebuild them for the new scene.
                    vision::MaterialLut::destroy_instance();
                    auto loaded=adapter->load_scene(active_scene);
                    if(!loaded.ok())throw std::runtime_error(loaded.error().message);
                }
                pipeline=lfs::denoise::make_pipeline(lane,calibration,geometry);
                lfs::backend::FrameProcessor* reconstruction=pipeline.get();
                if(resident){
                    resident_processor=std::make_unique<lfs::denoise::resident::ResidentAdapter>(*pipeline,*pipeline);
                    reconstruction=resident_processor.get();
                }
                processor=std::make_unique<lfs::lightfield::LightFieldProcessor>(calibration,geometry,reconstruction);
                if(!presenter.initialize({render_width,render_height},error))throw std::runtime_error(error);
                ++generation;frame=0;reset=true;error.clear();
            } catch(const std::exception &e) {processor.reset();resident_processor.reset();pipeline.reset();error=e.what();}
            rebuild=false;load=false;reload_work_scene=false;compiling=true;
        }
        if(processor&&!camera.paused) {
            camera.update(output,controller,dt);processor->set_camera(camera.camera);
            lfs::backend::FrameRequest request{};request.scene_path=active_scene;request.camera=camera.camera;
            request.sample_extent={render_width*3u,render_height};request.output_extent={render_width,render_height};
            request.sample_domain=lfs::backend::SampleDomain::EncodedSubpixelGrid;
            request.trace.spp=1;request.trace.max_depth=16;request.trace.geometry_layer_count=1;request.trace.require_external_primary_rays=true;
            request.trace.reuse_primary_hit=true;request.trace.geometry_motion_model=lfs::backend::TraceRequest::GeometryMotionModel::StaticOrRigid;
            request.trace.continuation_split_mode=lfs::backend::TraceRequest::ContinuationSplitMode::PrimaryResponsiveSpatiallyStratified;
            request.frame_index=++frame;request.scene_generation=generation;request.layout_generation=generation;request.layout_hash=generation;
            request.reset_reason=reset?lfs::backend::ResetReason::Camera:lfs::backend::ResetReason::None;reset=false;
            if(compiling)show_loading("Compiling rendering and reconstruction shaders...");
            auto rendered=adapter->render(request,processor.get());
            if(rendered.ok()) {
                auto view=adapter->presented_frame_view(rendered.value().completion_token);
                glfwMakeContextCurrent(output);
                if(view.ok()&&presenter.upload(view.value(),error)) {
                    int w,h;glfwGetFramebufferSize(output,&w,&h);glViewport(0,0,w,h);glClear(GL_COLOR_BUFFER_BIT);
                    presenter.draw(w,h);glfwSwapBuffers(output);error.clear();
                    const auto presented=std::chrono::steady_clock::now();
                    if (previous_presentation) {
                        const double seconds=std::chrono::duration<double>(presented-*previous_presentation).count();
                        if (seconds>0) fps=1./seconds;
                    }
                    previous_presentation=presented;
                    if(compiling) {
                        const auto filename=scene.filename().u8string();
                        logger.info("Scene {} ({})",reinterpret_cast<const char *>(filename.c_str()),display_modes[resolution].id);
                        logger.info("Studio presented {}x{}, {} views, {} paths",render_width,render_height,calibration.view_count,rendered.value().path_integration_count);
                        glfwSetWindowTitle(output,"LightFieldStudio - LFD output");compiling=false;
                    }
                } else if(!view.ok())error=view.error().message;
                frame_ms=rendered.value().frame_gpu_ms;
            } else {
                error=rendered.error().message;
                if(rendered.error().code==lfs::backend::ErrorCode::OutOfMemory)
                    error+=" Close other GPU applications or select the lower resolution, then resume.";
                camera.paused=true;
            }
        } else {
            previous_presentation.reset();
        }
        glfwMakeContextCurrent(controller);glfwSwapInterval(1);
        ImGui_ImplOpenGL3_NewFrame();ImGui_ImplGlfw_NewFrame();ImGui::NewFrame();
        int cw,ch;glfwGetFramebufferSize(controller,&cw,&ch);
        int logical_w,logical_h;glfwGetWindowSize(controller,&logical_w,&logical_h);
        ImGui::SetNextWindowPos({0,0});ImGui::SetNextWindowSize({float(logical_w),float(logical_h)});
        ImGui::Begin("LightFieldStudio",nullptr,ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoCollapse);
        ImGui::TextUnformatted("SSAT - Spatial-Angular-Temporal Reconstruction");
        const auto custom_name=scene.filename().u8string();
        const char *current_scene=selected_scene>=0 ? scenes[selected_scene].name :
            reinterpret_cast<const char *>(custom_name.c_str());
        if (ImGui::BeginCombo("Scene",current_scene)) {
            for (int i=0;i<static_cast<int>(scenes.size());++i) {
                if (ImGui::Selectable(scenes[i].name,selected_scene==i)) {
                    selected_scene=i;
                    scene=bundled_path(runtime,i,display_modes[resolution]);
                    load=true;rebuild=true;camera.paused=false;
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::Button("Open scene...")) {
            const auto selected=choose_scene(glfwGetWin32Window(controller));
            if (!selected.empty()) {
                selected_scene=-1;scene=selected;load=true;rebuild=true;camera.paused=false;
            }
        }
        ImGui::Separator();int quality=lane==lfs::denoise::Lane::Performance?1:0;
        if(ImGui::Combo("Quality",&quality,"Quality\0Performance\0")){lane=quality?lfs::denoise::Lane::Performance:lfs::denoise::Lane::Quality;rebuild=true;}
        if(ImGui::Combo("Resolution",&resolution,"3840 x 2160\0" "1536 x 2048\0" "7680 x 4320\0")) {
            const auto mode = display_modes[resolution];
            render_width=mode.width;render_height=mode.height;
            if (selected_scene>=0) {
                scene=bundled_path(runtime,selected_scene,mode);
                load=true;
            }
            if(auto *monitor=glfwGetWindowMonitor(output)) {
                const auto *native=glfwGetVideoMode(monitor);
                if(!native||native->width!=int(render_width)||native->height!=int(render_height))fullscreen=false;
            }
            reload_calibration=true;reload_work_scene=true;rebuild=true;camera.paused=false;
        }
        if(ImGui::Button("Reset camera")){camera.reset();reset=true;camera.paused=false;}
        ImGui::SameLine();if(ImGui::Button(camera.paused?"Resume":"Pause"))camera.paused=!camera.paused;
        ImGui::Separator();int count=0;auto monitors=glfwGetMonitors(&count);monitor_index=std::clamp(monitor_index,0,std::max(count-1,0));
        if(count&&ImGui::BeginCombo("Output display",glfwGetMonitorName(monitors[monitor_index]))) {
            for(int i=0;i<count;++i)if(ImGui::Selectable(glfwGetMonitorName(monitors[i]),i==monitor_index))monitor_index=i;
            ImGui::EndCombo();
        }
        bool current_fullscreen=glfwGetWindowMonitor(output)!=nullptr;
        if(fullscreen!=current_fullscreen || ImGui::Button(current_fullscreen?"Leave fullscreen":"Native fullscreen")) {
            settings_error.clear();
            if(fullscreen==current_fullscreen)fullscreen=!current_fullscreen;
            if(fullscreen&&count) {
                auto mode=glfwGetVideoMode(monitors[monitor_index]);
                if(mode->width!=int(render_width)||mode->height!=int(render_height)){settings_error="Set the output display to the selected native resolution first.";fullscreen=false;}
                else {
                    glfwSetWindowMonitor(output,monitors[monitor_index],0,0,mode->width,mode->height,mode->refreshRate);
                    int output_width=0,output_height=0;glfwGetFramebufferSize(output,&output_width,&output_height);
                    ocarina::logger().info("Native fullscreen framebuffer: {} x {}; render: {} x {}",output_width,output_height,render_width,render_height);
                }
            } else {fullscreen=false;glfwSetWindowMonitor(output,nullptr,100,100,1280,720,0);}
        }
        if(ImGui::CollapsingHeader("Light-field calibration")) {
            ImGui::TextWrapped("Use the calibration supplied with your physical display. Apply resets history.");
            int views=int(calibration.view_count);if(ImGui::InputInt("Views",&views))calibration.view_count=uint32_t(std::clamp(views,1,256));
            ImGui::InputDouble("Pitch",&calibration.pitch,0,0,"%.8f");
            ImGui::InputDouble("Slant (radians)",&calibration.slant,0,0,"%.8f");
            ImGui::InputDouble("Offset",&calibration.offset,0,0,"%.8f");
            ImGui::InputDouble("Focal distance",&geometry.focal_distance,0,0,"%.8f");
            ImGui::InputDouble("View angle (degrees)",&geometry.array_angle_degrees,0,0,"%.8f");
            if(ImGui::Button("Apply calibration")) {
                settings_error.clear();
                if(std::isfinite(calibration.pitch)&&calibration.pitch>0&&std::isfinite(calibration.slant)&&std::isfinite(calibration.offset)&&std::isfinite(geometry.focal_distance)&&geometry.focal_distance>0&&std::isfinite(geometry.array_angle_degrees)&&geometry.array_angle_degrees>=0&&geometry.array_angle_degrees<180)rebuild=true;
                else settings_error="Calibration values must be finite; pitch and focal distance must be positive, and view angle must be in [0, 180) degrees.";
            }
        }
        ImGui::Separator();ImGui::Text("GPU frame: %.2f ms | Display: %.1f FPS",frame_ms,fps);
        if(!error.empty())ImGui::TextWrapped("%s",error.c_str());
        if(!settings_error.empty())ImGui::TextWrapped("%s",settings_error.c_str());
        if(ImGui::CollapsingHeader("Controls"))ImGui::TextWrapped("Move with WASD and Q/E. Hold Shift to move faster, drag to look, and scroll to dolly. In the output window, F11 toggles fullscreen and Escape exits. Use native fullscreen on a calibrated LFD.");
        ImGui::End();ImGui::Render();glViewport(0,0,cw,ch);glClearColor(.035f,.04f,.05f,1);glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());glfwSwapBuffers(controller);
    }
    processor.reset();resident_processor.reset();pipeline.reset();
    vision::MaterialLut::destroy_instance();adapter.reset();
    glfwMakeContextCurrent(output);presenter.shutdown();
    glfwMakeContextCurrent(controller);ImGui_ImplOpenGL3_Shutdown();ImGui_ImplGlfw_Shutdown();ImGui::DestroyContext();
    glfwDestroyWindow(output);glfwDestroyWindow(controller);glfwTerminate();
    std::error_code ignored;std::filesystem::remove_all(session,ignored);return 0;
} catch(const std::exception &e) {
    ocarina::logger().error("{}",e.what());MessageBoxA(nullptr,e.what(),"LightFieldStudio",MB_OK|MB_ICONERROR);return 1;
}
