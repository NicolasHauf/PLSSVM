/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Implements custom exception classes derived from `std::runtime_error` including source location information.
 */

#pragma once

#include "plssvm/exceptions/source_location.hpp"  // plssvm::source_location

#include "fmt/core.h"  // fmt::format

#include <stdexcept>    // std::runtime_error
#include <string>       // std::string
#include <string_view>  // std::string_view

namespace plssvm {

/**
 * @brief Base class for all custom exception types. Forwards its message to [`std::runtime_error`](https://en.cppreference.com/w/cpp/error/runtime_error)
 *        and saves the call side source location information.
 */
class exception : public std::runtime_error {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message to [`std::runtime_error`](https://en.cppreference.com/w/cpp/error/runtime_error).
     * @param[in] msg the exception's `what()` message
     * @param[in] class_name the name of the thrown exception class
     * @param[in] loc the exception's call side information
     */
    explicit exception(const std::string &msg, const std::string_view class_name = "exception", source_location loc = source_location::current()) :
        std::runtime_error{ msg }, class_name_{ class_name }, loc_{ loc } {}

    /**
     * @brief Returns the information of the call side where the exception was thrown.
     * @return the exception's call side information (`[[nodiscard]]`)
     */
    [[nodiscard]] const source_location &loc() const noexcept { return loc_; }

    /**
     * @brief Returns a sting containing the exception's `what()` message, the name of the thrown exception class and information about the call
     *        side where the exception has been thrown.
     * @return the exception's `what()` message including source location information
     */
    [[nodiscard]] std::string what_with_loc() const {
        return fmt::format(
            "{}\n"
            "{} thrown:\n"
            "  in file      {}\n"
            "  in function  {}\n"
            "  @ line       {}",
            this->what(),
            class_name_,
            loc_.file_name(),
            loc_.function_name(),
            loc_.line());
    }

  private:
    const std::string_view class_name_;
    source_location loc_;
};

/**
 * @brief Exception type thrown if the provided data set file couldn't be found.
 */
class file_not_found_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to `plssvm::exception`.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit file_not_found_exception(const std::string &msg, source_location loc = source_location::current()) :
        exception{ msg, "file_not_found_exception", loc } {}
};

/**
 * @brief Exception type thrown if the provided data set file has an invalid format for the selected parser
 *        (e.g. if the arff parser tries to parse a libsvm file).
 */
class invalid_file_format_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to `plssvm::exception`.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit invalid_file_format_exception(const std::string &msg, source_location loc = source_location::current()) :
        exception{ msg, "invalid_file_format_exception", loc } {}
};

/**
 * @brief Exception type thrown if the requested backend is not supported on the target machine.
 */
class unsupported_backend_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to `plssvm::exception`.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit unsupported_backend_exception(const std::string &msg, source_location loc = source_location::current()) :
        exception{ msg, "unsupported_backend_exception", loc } {}
};

/**
 * @brief Exception type thrown if no data distribution between multiple devices could be created.
 */
class distribution_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to `plssvm::exception`.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit distribution_exception(const std::string &msg, source_location loc = source_location::current()) :
        exception{ msg, "distribution_exception", loc } {}
};

}  // namespace plssvm