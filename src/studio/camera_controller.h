#pragma once

#include "backend/frame_contract.h"
#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <cmath>

namespace lfs::studio {
using Vec3=std::array<float,3>;
inline Vec3 add(Vec3 a, Vec3 b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Vec3 sub(Vec3 a, Vec3 b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec3 mul(Vec3 a, float s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline float dot(Vec3 a, Vec3 b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
inline Vec3 normalize(Vec3 value) {
    const float length = std::sqrt(std::max(dot(value, value), 1.0e-12f));
    return mul(value, 1.f / length);
}

struct CameraController {
    lfs::backend::CameraState base{};
    lfs::backend::CameraState camera{};
    Vec3 dolly_target{};
    bool paused{};
    bool toggle_fullscreen_requested{};
    float yaw{};
    float pitch{};
    bool mouse_captured{};
    double last_mouse_x{};
    double last_mouse_y{};
    double pending_scroll_y{};
    double pending_mouse_x{}, pending_mouse_y{};
    bool mouse_buttons[2]{};

    void mouse_button(GLFWwindow *, int button, int action) {
        const int slot = button == GLFW_MOUSE_BUTTON_LEFT ? 0 :
            button == GLFW_MOUSE_BUTTON_RIGHT ? 1 : -1;
        if (slot < 0) return;
        // Preserve the queued cursor/button event order. A live cursor query
        // while processing an older press can cancel an already queued drag.
        mouse_buttons[slot] = action == GLFW_PRESS;
        mouse_captured = mouse_buttons[0] || mouse_buttons[1];
    }

    void cursor(double x, double y) {
        if (mouse_captured && !paused) {
            pending_mouse_x += x - last_mouse_x;
            pending_mouse_y += y - last_mouse_y;
        }
        last_mouse_x = x; last_mouse_y = y;
    }

    void reset() {
        camera = base;
        pending_mouse_x = pending_mouse_y = pending_scroll_y = 0;

        const Vec3 span = sub(base.target,base.position);
        const Vec3 forward = normalize(span);
        const Vec3 right = normalize(cross(forward,base.up));
        const Vec3 vertical = normalize(cross(right,forward));
        // Vision initializes its scene sensor through yaw and pitch in degrees.
        constexpr float pi = 3.14159265358979323846f;
        const float yaw_degrees = std::atan2(forward[0],right[0])*180.f/pi;
        const float pitch_degrees = std::atan2(vertical[2],vertical[1])*180.f/pi;
        yaw = yaw_degrees*pi/180.f;
        pitch = pitch_degrees*pi/180.f;
        const Vec3 direction{std::sin(yaw)*std::cos(pitch),std::sin(pitch),-std::cos(yaw)*std::cos(pitch)};
        camera.up = {-std::sin(yaw)*std::sin(pitch),std::cos(pitch),std::cos(yaw)*std::sin(pitch)};
        camera.target = add(camera.position,direction);
        dolly_target = add(camera.position,mul(direction,std::sqrt(dot(span,span))));
    }

    void add_scroll(double yoffset) noexcept {
        pending_scroll_y += yoffset;
    }

    bool update(
        GLFWwindow *render_window,
        GLFWwindow *control_window,
        float dt) {

        if (paused) return false;
        bool changed = false;
        const auto key_down = [&](int key) noexcept {
            return glfwGetKey(render_window, key) == GLFW_PRESS ||
                glfwGetKey(control_window, key) == GLFW_PRESS;
        };
        const float speed = key_down(GLFW_KEY_LEFT_SHIFT) ||
                key_down(GLFW_KEY_RIGHT_SHIFT)
            ? 2.0f : 0.7f;
        const Vec3 forward = normalize(sub(camera.target, camera.position));
        const Vec3 right = normalize(cross(forward, camera.up));
        Vec3 delta{};
        if (key_down(GLFW_KEY_W)) delta = add(delta, forward);
        if (key_down(GLFW_KEY_S)) delta = sub(delta, forward);
        if (key_down(GLFW_KEY_D)) delta = add(delta, right);
        if (key_down(GLFW_KEY_A)) delta = sub(delta, right);
        if (key_down(GLFW_KEY_E)) delta = add(delta, camera.up);
        if (key_down(GLFW_KEY_Q)) delta = sub(delta, camera.up);
        if (dot(delta, delta) > 1.0e-8f) {
            delta = mul(normalize(delta), speed * dt);
            camera.position = add(camera.position, delta);
            camera.target = add(camera.target, delta);
            dolly_target = add(dolly_target,delta);
            changed = true;
        }
        if (std::abs(pending_scroll_y) > 1.0e-6) {
            const float distance = std::sqrt(std::max(
                dot(sub(dolly_target, camera.position),
                    sub(dolly_target, camera.position)),
                1.0e-6f));
            const float dolly = std::clamp(
                static_cast<float>(pending_scroll_y) *
                    std::max(0.08f * distance, 0.05f),
                -0.8f * distance,
                0.8f * distance);
            camera.position = add(camera.position, mul(forward, dolly));
            camera.target = add(camera.position,forward);
            pending_scroll_y = 0.0;
            changed = true;
        }
        if (pending_mouse_x != 0.0 || pending_mouse_y != 0.0) {
            yaw += static_cast<float>(pending_mouse_x) * 0.0025f;
            pitch = std::clamp(pitch - static_cast<float>(pending_mouse_y) * 0.0025f,
                               -1.45f, 1.45f);
            pending_mouse_x = pending_mouse_y = 0.0;
            const Vec3 direction{
                std::sin(yaw) * std::cos(pitch), std::sin(pitch),
                -std::cos(yaw) * std::cos(pitch)};
            camera.target = add(camera.position, direction);
            camera.up = {-std::sin(yaw)*std::sin(pitch),std::cos(pitch),std::cos(yaw)*std::sin(pitch)};
            dolly_target = camera.target;
            changed = true;
        }
        return changed;
    }
};

} // namespace lfs::studio
