#pragma once

#include <hdf5.h>

#include <cstdint>
#include <filesystem>
#include <highfive/highfive.hpp>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/h5md/h5md_writer.hpp"

namespace SpongeH5MD
{
class HighFiveBackend : public WriterBackend
{
   public:
    bool Open(const WriterOptions& options) override
    {
        options_ = options;
        last_error_.clear();
        try
        {
            const std::filesystem::path file_path(options.path);
            const std::filesystem::path parent = file_path.parent_path();
            if (!parent.empty())
            {
                std::filesystem::create_directories(parent);
            }
            file_.reset(
                new HighFive::File(options.path, HighFive::File::Overwrite));
            status_ = FileStatus::open;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to open HDF5 file: ") + err.what());
        }
    }

    bool Flush() override
    {
        if (!Ensure_File()) return false;
        try
        {
            file_->flush();
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to flush HDF5 file: ") +
                        err.what());
        }
    }

    bool Close() override
    {
        if (file_ == nullptr)
        {
            status_ = FileStatus::closed;
            return true;
        }
        try
        {
            file_->flush();
            file_.reset();
            dataset_specs_.clear();
            status_ = FileStatus::closed;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to close HDF5 file: ") +
                        err.what());
        }
    }

    bool Finalize() override
    {
        if (!Ensure_File()) return false;
        if (!Write_String(path::output_status, "finalized")) return false;
        status_ = FileStatus::finalized;
        return Flush();
    }

    bool Ensure_Group(const std::string& group_path) override
    {
        if (!Ensure_File()) return false;
        if (group_path.empty() || group_path == "/") return true;
        try
        {
            std::string current;
            for (const std::string& component : Split_Path(group_path))
            {
                current += "/" + component;
                if (!file_->exist(current))
                {
                    file_->createGroup(current);
                }
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to ensure group ") + group_path +
                        ": " + err.what());
        }
    }

    bool Create_Dataset(const DatasetSpec& spec) override
    {
        if (!Ensure_File()) return false;
        if (spec.path.empty() || spec.shape.dims.empty())
        {
            return Fail("dataset path and dimensions must be non-empty");
        }
        try
        {
            if (!Ensure_Parent_Group(spec.path)) return false;
            if (file_->exist(spec.path))
            {
                dataset_specs_[spec.path] = Normalize_Spec(spec);
                return true;
            }
            const DatasetSpec normalized = Normalize_Spec(spec);
            const std::vector<hsize_t> dims = To_HSize(normalized.shape.dims);
            const std::vector<hsize_t> max_dims = To_HSize_Max(normalized);
            const std::vector<hsize_t> chunk_dims = To_HSize_Chunk(normalized);
            HighFive::DataSpace space(dims, max_dims);
            HighFive::DataSetCreateProps props;
            props.add(HighFive::Chunking(chunk_dims));
            switch (normalized.type)
            {
                case DataType::int64:
                    file_->createDataSet<int64_t>(spec.path, space, props);
                    break;
                case DataType::float32:
                    file_->createDataSet<float>(spec.path, space, props);
                    break;
                case DataType::float64:
                    file_->createDataSet<double>(spec.path, space, props);
                    break;
                case DataType::string:
                    file_->createDataSet<std::string>(spec.path, space, props);
                    break;
            }
            dataset_specs_[spec.path] = normalized;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to create dataset ") + spec.path +
                        ": " + err.what());
        }
    }

    bool Create_Virtual_Dataset(
        const DatasetSpec& spec,
        const std::vector<VirtualDatasetSource>& sources) override
    {
        if (!Ensure_File()) return false;
        if (spec.path.empty() || spec.shape.dims.empty())
        {
            return Fail(
                "virtual dataset path and dimensions must be non-empty");
        }
        if (sources.empty())
        {
            return Fail("virtual dataset sources must be non-empty: " +
                        spec.path);
        }
        if (spec.type == DataType::string)
        {
            return Fail("string virtual datasets are not supported: " +
                        spec.path);
        }
        try
        {
            if (!Ensure_Parent_Group(spec.path)) return false;
            Delete_If_Exists(spec.path);

            const DatasetSpec normalized = Normalize_Spec(spec);
            const std::vector<hsize_t> dims = To_HSize(normalized.shape.dims);
            hid_t virtual_space = H5Screate_simple(
                static_cast<int>(dims.size()), dims.data(), nullptr);
            if (virtual_space < 0)
            {
                return Fail("failed to create virtual dataspace: " + spec.path);
            }

            hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
            if (dcpl < 0)
            {
                H5Sclose(virtual_space);
                return Fail("failed to create virtual dataset property list: " +
                            spec.path);
            }

            for (const auto& source : sources)
            {
                if (source.source_dims.size() != normalized.shape.dims.size() ||
                    source.virtual_start.size() != normalized.shape.dims.size())
                {
                    H5Pclose(dcpl);
                    H5Sclose(virtual_space);
                    return Fail("virtual dataset source rank mismatch: " +
                                spec.path);
                }
                const std::vector<hsize_t> source_dims =
                    To_HSize(source.source_dims);
                const std::vector<hsize_t> virtual_start =
                    To_HSize(source.virtual_start);
                hid_t source_space =
                    H5Screate_simple(static_cast<int>(source_dims.size()),
                                     source_dims.data(), nullptr);
                if (source_space < 0)
                {
                    H5Pclose(dcpl);
                    H5Sclose(virtual_space);
                    return Fail("failed to create source dataspace for VDS: " +
                                spec.path);
                }
                const herr_t select_rc = H5Sselect_hyperslab(
                    virtual_space, H5S_SELECT_SET, virtual_start.data(),
                    nullptr, source_dims.data(), nullptr);
                if (select_rc < 0)
                {
                    H5Sclose(source_space);
                    H5Pclose(dcpl);
                    H5Sclose(virtual_space);
                    return Fail("failed to select VDS hyperslab: " + spec.path);
                }
                const herr_t virtual_rc = H5Pset_virtual(
                    dcpl, virtual_space, source.file_path.c_str(),
                    source.dataset_path.c_str(), source_space);
                H5Sclose(source_space);
                if (virtual_rc < 0)
                {
                    H5Pclose(dcpl);
                    H5Sclose(virtual_space);
                    return Fail("failed to map VDS source " + source.file_path +
                                ":" + source.dataset_path + " into " +
                                spec.path);
                }
            }

            const hid_t dataset = H5Dcreate2(
                file_->getId(), spec.path.c_str(), H5_Type(normalized.type),
                virtual_space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Pclose(dcpl);
            H5Sclose(virtual_space);
            if (dataset < 0)
            {
                return Fail("failed to create virtual dataset: " + spec.path);
            }
            H5Dclose(dataset);
            dataset_specs_[spec.path] = normalized;
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to create virtual dataset ") +
                        spec.path + ": " + err.what());
        }
    }

    bool Create_Hard_Link(const std::string& target,
                          const std::string& link_path) override
    {
        if (!Ensure_File()) return false;
        try
        {
            if (!Ensure_Parent_Group(link_path)) return false;
            if (file_->exist(link_path)) return true;
            if (!file_->exist(target))
            {
                return Fail("hard-link target does not exist: " + target);
            }
            const herr_t rc =
                H5Lcreate_hard(file_->getId(), target.c_str(), file_->getId(),
                               link_path.c_str(), H5P_DEFAULT, H5P_DEFAULT);
            if (rc < 0)
            {
                return Fail("failed to create hard link " + link_path + " -> " +
                            target);
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to create hard link ") + link_path +
                        ": " + err.what());
        }
    }

    bool Append_Int64(const std::string& dataset_path, const int64_t* data,
                      std::size_t count) override
    {
        return Append_Data(dataset_path, data, count);
    }

    bool Append_Float32(const std::string& dataset_path, const float* data,
                        std::size_t count) override
    {
        return Append_Data(dataset_path, data, count);
    }

    bool Append_Float64(const std::string& dataset_path, const double* data,
                        std::size_t count) override
    {
        return Append_Data(dataset_path, data, count);
    }

    bool Write_String(const std::string& dataset_path,
                      const std::string& value) override
    {
        if (!Ensure_File()) return false;
        try
        {
            if (!Ensure_Parent_Group(dataset_path)) return false;
            Delete_If_Exists(dataset_path);
            HighFive::DataSet dataset = file_->createDataSet<std::string>(
                dataset_path, HighFive::DataSpace::From(value));
            dataset.write(value);
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to write string dataset ") +
                        dataset_path + ": " + err.what());
        }
    }

    bool Write_String_Array(const std::string& dataset_path,
                            const std::vector<std::string>& values) override
    {
        if (!Ensure_File()) return false;
        try
        {
            if (!Ensure_Parent_Group(dataset_path)) return false;
            Delete_If_Exists(dataset_path);
            HighFive::DataSpace space({values.size()});
            HighFive::DataSet dataset =
                file_->createDataSet<std::string>(dataset_path, space);
            dataset.write(values);
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to write string-array dataset ") +
                        dataset_path + ": " + err.what());
        }
    }

    bool Set_Status(FileStatus status) override
    {
        status_ = status;
        return Write_String(path::output_status, Status_String(status));
    }

    FileStatus Status() const override { return status_; }
    std::string Last_Error() const override { return last_error_; }

   private:
    static hid_t H5_Type(const DataType type)
    {
        switch (type)
        {
            case DataType::int64:
                return H5T_NATIVE_LLONG;
            case DataType::float32:
                return H5T_NATIVE_FLOAT;
            case DataType::float64:
                return H5T_NATIVE_DOUBLE;
            case DataType::string:
                return H5T_C_S1;
        }
        return H5T_NATIVE_DOUBLE;
    }

    template <typename T>
    bool Append_Data(const std::string& dataset_path, const T* data,
                     std::size_t count)
    {
        if (!Ensure_File()) return false;
        if (data == nullptr)
        {
            return Fail("append data pointer is null for dataset " +
                        dataset_path);
        }
        try
        {
            const auto spec_iter = dataset_specs_.find(dataset_path);
            if (spec_iter == dataset_specs_.end() ||
                !file_->exist(dataset_path))
            {
                return Fail("dataset was not created before append: " +
                            dataset_path);
            }
            HighFive::DataSet dataset = file_->getDataSet(dataset_path);
            std::vector<std::size_t> dims = dataset.getSpace().getDimensions();
            if (dims.empty())
            {
                return Fail("append target has scalar dataspace: " +
                            dataset_path);
            }
            const std::size_t record_size = Record_Size(dims);
            std::size_t append_records = 1;
            std::size_t expected_count = record_size;
            const bool fixed_first_dim_bulk_append =
                spec_iter->second.appendable &&
                !spec_iter->second.shape.max_dims.empty() &&
                spec_iter->second.shape.max_dims[0] != 0;
            if (fixed_first_dim_bulk_append && record_size > 0 &&
                count % record_size == 0)
            {
                append_records = count / record_size;
                expected_count = count;
            }
            if (count != expected_count || append_records == 0)
            {
                return Fail(
                    "append count does not match one dataset record for " +
                    dataset_path);
            }
            std::vector<std::size_t> new_dims = dims;
            new_dims[0] += append_records;
            dataset.resize(new_dims);
            std::vector<std::size_t> offset(dims.size(), 0);
            offset[0] = dims[0];
            std::vector<std::size_t> selection_count = new_dims;
            selection_count[0] = append_records;
            for (std::size_t i = 1; i < selection_count.size(); ++i)
            {
                selection_count[i] = dims[i];
            }
            const std::vector<hsize_t> h_offset = To_HSize(offset);
            const std::vector<hsize_t> h_count = To_HSize(selection_count);
            hid_t file_space = H5Dget_space(dataset.getId());
            if (file_space < 0)
            {
                return Fail("failed to get append dataspace: " + dataset_path);
            }
            const herr_t select_rc =
                H5Sselect_hyperslab(file_space, H5S_SELECT_SET, h_offset.data(),
                                    nullptr, h_count.data(), nullptr);
            if (select_rc < 0)
            {
                H5Sclose(file_space);
                return Fail("failed to select append hyperslab: " +
                            dataset_path);
            }
            hid_t mem_space = H5Screate_simple(static_cast<int>(h_count.size()),
                                               h_count.data(), nullptr);
            if (mem_space < 0)
            {
                H5Sclose(file_space);
                return Fail("failed to create append memory dataspace: " +
                            dataset_path);
            }
            const herr_t write_rc =
                H5Dwrite(dataset.getId(), H5_Type(spec_iter->second.type),
                         mem_space, file_space, H5P_DEFAULT, data);
            H5Sclose(mem_space);
            H5Sclose(file_space);
            if (write_rc < 0)
            {
                return Fail("failed to write append hyperslab: " +
                            dataset_path);
            }
            return true;
        }
        catch (const std::exception& err)
        {
            return Fail(std::string("failed to append dataset ") +
                        dataset_path + ": " + err.what());
        }
    }

    bool Ensure_File()
    {
        if (file_ != nullptr) return true;
        return Fail("HDF5 file is not open");
    }

    bool Ensure_Parent_Group(const std::string& object_path)
    {
        const std::size_t pos = object_path.find_last_of('/');
        if (pos == std::string::npos || pos == 0) return true;
        return Ensure_Group(object_path.substr(0, pos));
    }

    void Delete_If_Exists(const std::string& object_path)
    {
        if (file_->exist(object_path))
        {
            H5Ldelete(file_->getId(), object_path.c_str(), H5P_DEFAULT);
            dataset_specs_.erase(object_path);
        }
    }

    DatasetSpec Normalize_Spec(const DatasetSpec& spec) const
    {
        DatasetSpec normalized = spec;
        if (normalized.shape.max_dims.size() != normalized.shape.dims.size())
        {
            normalized.shape.max_dims.assign(normalized.shape.dims.size(), 0);
        }
        if (normalized.shape.chunk_dims.size() != normalized.shape.dims.size())
        {
            normalized.shape.chunk_dims = normalized.shape.dims;
        }
        for (std::size_t i = 0; i < normalized.shape.chunk_dims.size(); ++i)
        {
            if (normalized.shape.chunk_dims[i] == 0)
            {
                normalized.shape.chunk_dims[i] = 1;
            }
        }
        return normalized;
    }

    std::vector<hsize_t> To_HSize(const std::vector<std::size_t>& values) const
    {
        std::vector<hsize_t> converted;
        converted.reserve(values.size());
        for (std::size_t value : values)
        {
            converted.push_back(static_cast<hsize_t>(value));
        }
        return converted;
    }

    std::vector<hsize_t> To_HSize_Max(const DatasetSpec& spec) const
    {
        std::vector<hsize_t> converted;
        converted.reserve(spec.shape.dims.size());
        for (std::size_t i = 0; i < spec.shape.dims.size(); ++i)
        {
            const std::size_t configured = spec.shape.max_dims[i];
            if (spec.appendable && i == 0 && configured == 0)
            {
                converted.push_back(H5S_UNLIMITED);
            }
            else if (configured == 0)
            {
                converted.push_back(static_cast<hsize_t>(spec.shape.dims[i]));
            }
            else
            {
                converted.push_back(static_cast<hsize_t>(configured));
            }
        }
        return converted;
    }

    std::vector<hsize_t> To_HSize_Chunk(const DatasetSpec& spec) const
    {
        return To_HSize(spec.shape.chunk_dims);
    }

    static std::size_t Record_Size(const std::vector<std::size_t>& dims)
    {
        if (dims.size() == 1) return 1;
        return std::accumulate(
            dims.begin() + 1, dims.end(), static_cast<std::size_t>(1),
            [](std::size_t lhs, std::size_t rhs) { return lhs * rhs; });
    }

    static std::vector<std::string> Split_Path(const std::string& path)
    {
        std::vector<std::string> components;
        std::size_t start = 0;
        while (start < path.size())
        {
            while (start < path.size() && path[start] == '/') ++start;
            if (start >= path.size()) break;
            const std::size_t end = path.find('/', start);
            components.push_back(path.substr(start, end == std::string::npos
                                                        ? std::string::npos
                                                        : end - start));
            if (end == std::string::npos) break;
            start = end + 1;
        }
        return components;
    }

    static const char* Status_String(FileStatus status)
    {
        switch (status)
        {
            case FileStatus::closed:
                return "closed";
            case FileStatus::open:
                return "open";
            case FileStatus::closing:
                return "closing";
            case FileStatus::finalized:
                return "finalized";
            case FileStatus::failed:
                return "failed";
        }
        return "unknown";
    }

    bool Fail(const std::string& message)
    {
        last_error_ = message;
        status_ = FileStatus::failed;
        return false;
    }

    WriterOptions options_;
    std::unique_ptr<HighFive::File> file_;
    std::unordered_map<std::string, DatasetSpec> dataset_specs_;
    FileStatus status_ = FileStatus::closed;
    std::string last_error_;
};

class HighFiveBackendFactory : public WriterBackendFactory
{
   public:
    std::unique_ptr<WriterBackend> Create_Backend() override
    {
        return std::unique_ptr<WriterBackend>(new HighFiveBackend());
    }
};
}  // namespace SpongeH5MD
