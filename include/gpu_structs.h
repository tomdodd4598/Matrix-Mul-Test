#ifndef GPU_STRUCTS_H
#define GPU_STRUCTS_H

#include "slice.h"
#include "type_utils.h"

#include <array>
#include <utility>
#include <vector>
#include <type_traits>

template<typename T>
class Data {
    T* m_ptr;
    usize m_size;

public:
    typedef std::vector<T> Host;

    Data() = delete;

    Data(usize size) : m_ptr{nullptr}, m_size{size} {
        cudaMalloc(&m_ptr, bytes());
    }

    Data(usize size, u8 value) : Data(size) {
        memset(value);
    }

    Data(T const* h_ptr, usize size) : Data(size) {
        copy_from_host(h_ptr);
    }

    Data(Slice<T> slice) : Data(slice.ptr, slice.len) {}

    Data(SliceMut<T> slice) : Data(slice.ptr, slice.len) {}

    template<typename VECTOR, std::enable_if_t<is_class_v<remove_cvref_t<VECTOR>>, int> = 0>
    Data(VECTOR&& vec) : Data(vec.data(), vec.size()) {}

    Data(Data const& other) = delete;

    Data(Data&& other) : m_ptr{std::exchange(other.m_ptr, nullptr)}, m_size{std::exchange(other.m_size, 0)} {}

    ~Data() {
        cudaFree(m_ptr);
    }

    Data& operator=(Data const& other) = delete;

    Data& operator=(Data&& other) {
        if (this != &other) {
            cudaFree(m_ptr);
            m_ptr = std::exchange(other.m_ptr, nullptr);
            m_size = std::exchange(other.m_size, 0);
        }
        return *this;
    }

    operator SliceMut<T>() const {
        return SliceMut<T>(m_ptr, m_size);
    }

    T* ptr() const {
        return m_ptr;
    }

    usize size() const {
        return m_size;
    }

    usize bytes() const {
        return m_size * sizeof(T);
    }

    void memset(u8 value) {
        cudaMemset(m_ptr, value, bytes());
    }

    void copy_from_host(T const* h_ptr) {
        cudaMemcpy(m_ptr, h_ptr, bytes(), cudaMemcpyHostToDevice);
    }

    template<typename VECTOR>
    void copy_from_host(VECTOR&& vec) {
        copy_from_host(vec.data());
    }

    void copy_to_host(T* h_ptr) const {
        cudaMemcpy(h_ptr, m_ptr, bytes(), cudaMemcpyDeviceToHost);
    }

    void copy_to_host(Host& vec) const {
        if (vec.size() != m_size) {
            vec.resize(m_size);
        }
        copy_to_host(vec.data());
    }

    Host copy_to_host() const {
        Host vec(m_size);
        copy_to_host(vec);
        return vec;
    }
};

template<typename T>
class Data<T*> {
    T** m_ptr;
    usize m_size;
    std::vector<Data<T>> owned;

public:
    typedef std::vector<typename Data<T>::Host> Host;

    Data() = delete;

    template<typename VECTOR>
    Data(VECTOR&& vec) : m_ptr{nullptr}, m_size{vec.size()} {
        cudaMalloc(&m_ptr, bytes());

        std::vector<T*> owned_ptrs;
        for (auto const& e : vec) {
            Data<T> data(e);
            T* p = data.ptr();
            owned.push_back(std::move(data));
            owned_ptrs.push_back(p);
        }

        cudaMemcpy(m_ptr, owned_ptrs.data(), bytes(), cudaMemcpyHostToDevice);
    }

    Data(Data const& other) = delete;

    Data(Data&& other) : m_ptr{std::exchange(other.m_ptr, nullptr)}, m_size{std::exchange(other.m_size, 0)}, owned{std::move(other.owned)} {}

    ~Data() {
        cudaFree(m_ptr);
    }

    Data& operator=(Data const& other) = delete;

    Data& operator=(Data&& other) {
        if (this != &other) {
            cudaFree(m_ptr);
            m_ptr = std::exchange(other.m_ptr, nullptr);
            m_size = std::exchange(other.m_size, 0);
            owned = std::move(other.owned);
        }
        return *this;
    }

    operator SliceMut<T*>() const {
        return SliceMut<T*>(m_ptr, m_size);
    }

    T** ptr() const {
        return m_ptr;
    }

    usize size() const {
        return m_size;
    }

    usize bytes() const {
        return m_size * sizeof(T*);
    }

    void memset(u8 value) {
        for (auto& e : owned) {
            e.memset(value);
        }
    }

    template<typename VECTOR>
    void copy_from_host(VECTOR&& vec) {
        for (usize i = 0; i < m_size; ++i) {
            owned[i].copy_from_host(vec[i]);
        }
    }

    void copy_to_host(Host& vec) const {
        if (vec.size() != m_size) {
            vec.resize(m_size);
        }
        for (usize i = 0; i < m_size; ++i) {
            owned[i].copy_to_host(vec[i]);
        }
    }

    Host copy_to_host() const {
        Host vec(m_size);
        copy_to_host(vec);
        return vec;
    }
};

#endif
