#pragma once

#include "backend/frame_contract.h"
#include <glad.h>
#include <cuda.h>
#include <cudaGL.h>
#include <string>

namespace lfs::studio {
class InteropPresenter final {
private:
    backend::Extent2D extent_{};
    GLuint textures_[2]{};
    CUgraphicsResource resources_[2]{};
    uint32_t index_{};
    CUdevice device_{};
    CUcontext context_{};
    bool context_current_{};

public:
    bool initialize(backend::Extent2D extent, std::string &error) {
        extent_ = extent;
        CUresult context_status = cuInit(0u);
        if (context_status == CUDA_SUCCESS) {
            unsigned count=0;CUdevice devices[8]{};
            context_status=cuGLGetDevices(&count,devices,8,CU_GL_DEVICE_LIST_ALL);
            if(context_status==CUDA_SUCCESS && (!count || devices[0]!=0)) {
                error="Select the NVIDIA GPU for LightFieldStudio in Windows Graphics settings.";
                return false;
            }
            if(context_status==CUDA_SUCCESS)device_=devices[0];
        }
        if (context_status == CUDA_SUCCESS) {
            context_status = cuDevicePrimaryCtxRetain(&context_, device_);
        }
        if (context_status == CUDA_SUCCESS) {
            context_status = cuCtxPushCurrent(context_);
            context_current_ = context_status == CUDA_SUCCESS;
        }
        if (context_status != CUDA_SUCCESS) {
            error = "The renderer CUDA primary context could not be made current: " +
                std::to_string(static_cast<int>(context_status));
            return false;
        }
        glGenTextures(2, textures_);
        for (uint32_t i = 0; i < 2u; ++i) {
            glBindTexture(GL_TEXTURE_2D, textures_[i]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, extent_.width, extent_.height, 0,
                         GL_RGBA, GL_FLOAT, nullptr);
            const CUresult status = cuGraphicsGLRegisterImage(
                &resources_[i], textures_[i], GL_TEXTURE_2D,
                CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);
            if (status != CUDA_SUCCESS) {
                error = "cuGraphicsGLRegisterImage failed: " +
                    std::to_string(static_cast<int>(status));
                return false;
            }
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        return true;
    }

    bool upload(const lfs::backend::PresentedFrameView &frame, std::string &error) {
        if (frame.extent != lfs::backend::Extent2D{extent_.width, extent_.height} ||
            frame.rgba32f.size() != static_cast<size_t>(extent_.width) * extent_.height) {
            error = "Presented GPU frame has the wrong extent.";
            return false;
        }
        index_ ^= 1u;
        CUresult status = cuGraphicsMapResources(1, &resources_[index_], nullptr);
        CUarray array{};
        if (status == CUDA_SUCCESS) {
            status = cuGraphicsSubResourceGetMappedArray(
                &array, resources_[index_], 0u, 0u);
        }
        if (status == CUDA_SUCCESS) {
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.srcDevice = static_cast<CUdeviceptr>(frame.rgba32f.handle()) +
                frame.rgba32f.offset_in_byte();
            copy.srcPitch = static_cast<size_t>(extent_.width) * sizeof(ocarina::float4);
            copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            copy.dstArray = array;
            copy.WidthInBytes = copy.srcPitch;
            copy.Height = extent_.height;
            status = cuMemcpy2D(&copy);
        }
        const CUresult unmap = cuGraphicsUnmapResources(
            1, &resources_[index_], nullptr);
        if (status == CUDA_SUCCESS) status = unmap;
        if (status != CUDA_SUCCESS) {
            error = "CUDA/OpenGL presentation copy failed: " +
                std::to_string(static_cast<int>(status));
            return false;
        }
        return true;
    }

    void draw(int width, int height) const {
        GLuint fbo = 0;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(
            GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
            textures_[index_], 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        {
            // Canonical upright mapping from Vision's bottom-up framebuffer.
            glBlitFramebuffer(
                0, extent_.height, extent_.width, 0, 0, 0, width, height,
                GL_COLOR_BUFFER_BIT, GL_NEAREST);
        }
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
    }

    void shutdown() {
        for (uint32_t i = 0; i < 2u; ++i) {
            if (resources_[i]) {
                cuGraphicsUnregisterResource(resources_[i]);
                resources_[i] = {};
            }
        }
        if (textures_[0] || textures_[1]) {
            glDeleteTextures(2, textures_);
            textures_[0] = textures_[1] = 0u;
        }
        if (context_current_) {
            CUcontext popped{};
            cuCtxPopCurrent(&popped);
            context_current_ = false;
        }
        if (context_) {
            cuDevicePrimaryCtxRelease(device_);
            context_ = {};
        }
    }

    ~InteropPresenter() { shutdown(); }
};

} // namespace lfs::studio
